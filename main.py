import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, BinaryIO
import httpx
from jose import JWTError, jwt
from starlette.middleware.sessions import SessionMiddleware
import csv
import io
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")


# Add this after creating the FastAPI app
app = FastAPI(title="Objaverse Research Portal")
# Add this line:
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# API endpoints
API_URL = os.getenv("API_URL", "http://localhost:3000")
AUTH_URL = os.getenv("AUTH_URL", "http://localhost:4000")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

# Models
class User(BaseModel):
    userId: str
    email: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Authentication dependency
async def get_current_user(request: Request):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Get token from session
    token = request.session.get("access_token")

    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")    

    if not token:
        raise credentials_exception
    
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        userId: str = payload.get("sub")
        if userId is None:
            return None
        
        # Get user info from auth service
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(f"{AUTH_URL}/me", headers=headers)
            if response.status_code != 200:
                return None
            user_data = response.json()
            return User(**user_data)
    except JWTError:
        return None
    except Exception:
        return None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get the token explicitly
    token = request.session.get('access_token')
    if not token:
        return RedirectResponse(url="/login")
    
    # Get assigned models for evaluation
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/assignments?userId={user.userId}", headers=headers)
        if response.status_code != 200:
            assignments = []
        else:
            assignments = response.json().get("data", [])
    
    # Get completed evaluations count
    completed_count = 0
    avg_rating = 0
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.post(f"{API_URL}/api/completed", json={"userId": user.userId}, headers=headers)
        if response.status_code == 200:
            completed_data = response.json().get("data", [])
            completed_count = len(completed_data)
            
            # Calculate average rating if there are completed evaluations
            if completed_count > 0:
                total_ratings = 0
                rating_count = 0
                
                for eval_data in completed_data:
                    if "ratings" in eval_data:
                        for rating_key, rating_value in eval_data["ratings"].items():
                            total_ratings += rating_value
                            rating_count += 1
                
                if rating_count > 0:
                    avg_rating = round(total_ratings / rating_count, 1)
    
    # Get recent completed evaluations (limited to 3)
    recent_completed = []
    
    if completed_count > 0:
        # Sort by completion date (newest first) and take first 3
        sorted_completed = sorted(completed_data, key=lambda x: x.get("completedAt", ""), reverse=True)
        recent_completed = sorted_completed[:3]
    
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request, 
            "user": user, 
            "assignments": assignments,
            "pending_count": len(assignments),
            "completed_count": completed_count,
            "avg_rating": avg_rating,
            "recent_completed": recent_completed
        }
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    # Call auth service to get token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_URL}/login",
                data={"username": email, "password": password}
            )
            
            if response.status_code != 200:
                return templates.TemplateResponse(
                    "login.html",
                    {"request": request, "error": "Invalid email or password"}
                )
            
            token_data = response.json()
            
            # Store token in session
            request.session["access_token"] = token_data["access_token"]
            
            return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": str(e)}
        )

@app.get("/logout")
async def logout(request: Request):
    # Remove token from session
    request.session.pop("access_token", None)
    
    return RedirectResponse(url="/login")

@app.get("/evaluate/{object_id}", response_class=HTMLResponse)
async def evaluate_page(request: Request, object_id: str, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get object details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/objects/{object_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Object not found")
        
        object_data = response.json().get("data", {})
    
    return templates.TemplateResponse(
        "evaluate.html",
        {"request": request, "user": user, "object": object_data}
    )

@app.post("/evaluate/{object_id}")
async def submit_evaluation(
    request: Request,
    object_id: str,
    accuracy: int = Form(...),
    completeness: int = Form(...),
    clarity: int = Form(...),
    comments: str = Form(""),
    user: Optional[User] = Depends(get_current_user)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Submit evaluation to API
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        evaluation_data = {
            "objectId": object_id,
            "userId": user.userId,
            "ratings": {
                "accuracy": accuracy,
                "completeness": completeness,
                "clarity": clarity
            },
            "comments": comments
        }
        
        response = await client.post(
            f"{API_URL}/api/evaluations",
            json=evaluation_data,
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            async with httpx.AsyncClient() as obj_client:
                obj_response = await obj_client.get(
                    f"{API_URL}/api/objects/{object_id}",
                    headers=headers
                )
                object_data = obj_response.json().get("data", {})
                
            return templates.TemplateResponse(
                "evaluate.html",
                {
                    "request": request,
                    "user": user,
                    "object": object_data,
                    "error": "Failed to submit evaluation"
                }
            )
    
    return RedirectResponse(url="/", status_code=303)


@app.get("/review/{object_id}", response_class=HTMLResponse)
async def review_object(request: Request, object_id: str, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get object details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/objects/{object_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Object not found")
        
        object_data = response.json().get("data", {})
    
    return templates.TemplateResponse(
        "review.html",
        {"request": request, "user": user, "object": object_data}
    )

@app.post("/review/{object_id}")
async def submit_review(
    request: Request,
    object_id: str,
    accuracy: int = Form(...),
    completeness: int = Form(...),
    clarity: int = Form(...),
    comments: str = Form(""),
    user: Optional[User] = Depends(get_current_user)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Submit review to API
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        review_data = {
            "objectId": object_id,
            "userId": user.userId,
            "ratings": {
                "accuracy": accuracy,
                "completeness": completeness,
                "clarity": clarity
            },
            "comments": comments
        }
        
        response = await client.post(
            f"{API_URL}/api/reviews",
            json=review_data,
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            async with httpx.AsyncClient() as obj_client:
                obj_response = await obj_client.get(
                    f"{API_URL}/api/objects/{object_id}",
                    headers=headers
                )
                object_data = obj_response.json().get("data", {})
                
            return templates.TemplateResponse(
                "review.html",
                {
                    "request": request,
                    "user": user,
                    "object": object_data,
                    "error": "Failed to submit review"
                }
            )
    
    return RedirectResponse(url="/", status_code=303)

# Assignments route to match the sidebar link
@app.get("/assignments", response_class=HTMLResponse)
async def assignments_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get assigned models for evaluation
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/assignments?userId={user.userId}", headers=headers)
        if response.status_code != 200:
            assignments = []
        else:
            assignments = response.json().get("data", [])
    
    return templates.TemplateResponse(
        "assignments.html",
        {"request": request, "user": user, "assignments": assignments}
    )

@app.get("/assignments/{assignment_id}", response_model=Dict[str, Any])
async def get_assignment(request: Request, assignment_id: str, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get assignment details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/assignments/{assignment_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        assignment = response.json().get("data", {})
    
    return assignment

# Completed evaluations page to match the sidebar link
@app.get("/completed", response_class=HTMLResponse)
async def completed_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get completed evaluations for the user
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.post(f"{API_URL}/api/completed", json={"userId": user.userId}, headers=headers)
        if response.status_code != 200:
            evaluations = []
        else:
            evaluations = response.json().get("data", [])
    
    return templates.TemplateResponse(
        "completed.html",
        {"request": request, "user": user, "evaluations": evaluations}
    )

@app.post("/completed", response_model=List[Dict[str, Any]])
async def get_completed_evaluations(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get completed evaluations for the user
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.post(f"{API_URL}/api/completed", json={"userId": user.userId}, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Completed evaluations not found")
        
        evaluations = response.json().get("data", [])
    
    return evaluations


@app.get("/completed/{evaluation_id}", response_model=Dict[str, Any])
async def get_completed_evaluation(request: Request, evaluation_id: str, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Get completed evaluation details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/completed/{evaluation_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Completed evaluation not found")
        
        evaluation = response.json().get("data", {})
    
    return evaluation

# Get current user profile
@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("profile.html", {"request": request, "user": user})


@app.post("/profile")
async def update_profile(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Update user profile in auth service
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.put(
            f"{AUTH_URL}/users/{user.userId}",
            json={"email": email, "password": password},
            headers=headers
        )
        
        if response.status_code != 200:
            return templates.TemplateResponse(
                "profile.html",
                {"request": request, "user": user, "error": "Failed to update profile"}
            )
    
    return RedirectResponse(url="/profile", status_code=303)


# Fixed admin page route decorator
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Get all users
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{AUTH_URL}/users", headers=headers)
        if response.status_code != 200:
            users = []
        else:
            users = response.json().get("data", [])
    
    return templates.TemplateResponse("admin.html", {"request": request, "user": user, "users": users})


# use the admin page to assign models to users and create new users
@app.post("/admin/assign")
async def assign_model(
    request: Request,
    user_id: str = Form(...),
    model_id: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Assign model to user
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.post(
            f"{API_URL}/api/assignments",
            json={"userId": user_id, "modelId": model_id},
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "user": user, "error": "Failed to assign model"}
            )
    
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/create_user")
async def create_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Create new user in auth service
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.post(
            f"{AUTH_URL}/register",
            json={"email": email, "password": password, "role": role},
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "user": user, "error": "Failed to create user"}
            )
    
    return RedirectResponse(url="/admin", status_code=303)

# New route for uploading CSV to assign models
@app.post("/admin/upload_assignments")
async def upload_assignments_csv(
    request: Request,
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Process CSV file
    assignments = []
    errors = []
    
    try:
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            if "userId" not in row or "modelId" not in row:
                errors.append(f"Row {csv_reader.line_num}: Missing required fields (userId, modelId)")
                continue
                
            # Create assignment
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
                response = await client.post(
                    f"{API_URL}/api/assignments",
                    json={"userId": row["userId"], "modelId": row["modelId"]},
                    headers=headers
                )
                
                if response.status_code != 200 and response.status_code != 201:
                    errors.append(f"Row {csv_reader.line_num}: Failed to create assignment")
                else:
                    assignments.append(row)
    
    except Exception as e:
        return templates.TemplateResponse(
            "admin.html",
            {
                "request": request, 
                "user": user, 
                "error": f"Failed to process CSV: {str(e)}",
                "users": []  # Need to fetch users again
            }
        )
    
    # Get all users to render admin page
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{AUTH_URL}/users", headers=headers)
        if response.status_code != 200:
            users = []
        else:
            users = response.json().get("data", [])
    
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request, 
            "user": user, 
            "users": users,
            "success_message": f"Successfully created {len(assignments)} assignments",
            "errors": errors
        }
    )

# Export users to CSV
@app.get("/admin/export_users")
async def export_users_csv(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Get all users
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{AUTH_URL}/users", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Users not found")
        
        users_data = response.json().get("data", [])
    
    # Create CSV
    output = io.StringIO()
    fieldnames = ["userId", "email", "role"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for user_item in users_data:
        writer.writerow({
            "userId": user_item.get("userId", ""),
            "email": user_item.get("email", ""),
            "role": user_item.get("role", "")
        })
    
    output.seek(0)
    
    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]), 
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=users.csv"
    
    return response

@app.get("/admin/edit_user/{user_id}", response_class=HTMLResponse)
async def edit_user_page(request: Request, user_id: str, user: Optional[User] = Depends(get_current_user)):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Get user details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{AUTH_URL}/users/{user_id}", headers=headers)
        if response.status_code != 200:
            return RedirectResponse(url="/admin", status_code=303)
        
        user_data = response.json()
    
    return templates.TemplateResponse(
        "edit_user.html",
        {"request": request, "user": user, "edit_user": user_data}
    )


@app.post("/admin/update_user/{user_id}")
async def update_user(
    request: Request,
    user_id: str,
    email: str = Form(...),
    password: Optional[str] = Form(None),
    role: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Prepare update data
    update_data = {"email": email, "role": role}
    
    # Only include password if it was provided
    if password and password.strip():
        update_data["password"] = password
    
    # Update user in auth service
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.put(
            f"{AUTH_URL}/register/{user_id}",
            json=update_data,
            headers=headers
        )
        
        if response.status_code != 200:
            # Get user details again to redisplay the form
            user_response = await client.get(
                f"{AUTH_URL}/users/{user_id}", 
                headers=headers
            )
            
            if user_response.status_code == 200:
                user_data = user_response.json()
            else:
                return RedirectResponse(url="/admin", status_code=303)
                
            return templates.TemplateResponse(
                "edit_user.html",
                {
                    "request": request, 
                    "user": user, 
                    "edit_user": user_data,
                    "error": "Failed to update user"
                }
            )
    
    return RedirectResponse(url="/admin?updated=true", status_code=303)

@app.post("/admin/delete_user")
async def delete_user(
    request: Request,
    user_id: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Prevent self-deletion
    if user_id == user.userId:
        # Get all users to render admin page
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
            users_response = await client.get(f"{AUTH_URL}/users", headers=headers)
            if users_response.status_code == 200:
                users = users_response.json().get("data", [])
            else:
                users = []
                
        return templates.TemplateResponse(
            "admin.html",
            {
                "request": request, 
                "user": user, 
                "users": users,
                "error": "You cannot delete your own account"
            }
        )
    
    # Delete user in auth service
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.delete(
            f"{AUTH_URL}/register/{user_id}",
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 204:
            # Get all users to render admin page again
            users_response = await client.get(f"{AUTH_URL}/users", headers=headers)
            if users_response.status_code == 200:
                users = users_response.json().get("data", [])
            else:
                users = []
                
            return templates.TemplateResponse(
                "admin.html",
                {
                    "request": request, 
                    "user": user, 
                    "users": users,
                    "error": "Failed to delete user"
                }
            )
    
    return RedirectResponse(url="/admin?deleted=true", status_code=303)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))