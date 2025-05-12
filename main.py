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
import json
import time
from fastapi import Request, status
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException, RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException



# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")


# Add this after creating the FastAPI app
app = FastAPI(title="Objaverse Research Portal")

# Add this line:
app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY,
    session_cookie="objaverse_session",
    max_age=3600  # 1 hour
)

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

# Function to fetch all assignments for a user
async def fetch_all_assignments(access_token, user_id):
    all_assignments = []
    page = 1
    total_pages = 1  # Start with 1, will be updated from first response
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Loop through all pages
        while page <= total_pages:
            response = await client.get(f"{API_URL}/api/assignments?userId={user_id}&page={page}", headers=headers)
            
            if response.status_code != 200:
                break
                
            response_data = response.json()
            page_assignments = response_data.get("data", [])
            
            # Process assignments
            for obj in page_assignments:
                assignment = {
                    "object": obj  # Wrap each object in an assignment structure
                }
                all_assignments.append(assignment)
            
            # Update total pages
            total_pages = response_data.get("pages", 1)
            page += 1
    
    return all_assignments


# Function to fetch all completed evaluations for a user
async def fetch_all_completed_evaluations(access_token, user_id):
    all_completed = []
    page = 1
    total_pages = 1  # Start with 1, will be updated from first response
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Loop through all pages
        while page <= total_pages:
            try:
                response = await client.post(
                    f"{API_URL}/api/completed?page={page}",
                    json={"userId": user_id},
                    headers=headers
                )
                
                if response.status_code != 200:
                    print(f"Error fetching completed evaluations: {response.status_code} - {response.text}")
                    break
                    
                response_data = response.json()
                completed_items = response_data.get("data", [])
                all_completed.extend(completed_items)
                
                # Update total pages - ensure it's at least the current page if not provided
                total_pages = max(response_data.get("pages", page), page)
                
                # Debug information
                print(f"Fetched page {page}/{total_pages} of completed evaluations. Items: {len(completed_items)}")
                
                # Break if no items were returned, even if pages suggest more
                if not completed_items:
                    print(f"No items returned for page {page}, stopping pagination.")
                    break
                    
                page += 1
            except Exception as e:
                print(f"Exception fetching completed evaluations: {str(e)}")
                break
    
    print(f"Total completed evaluations fetched: {len(all_completed)}")
    return all_completed

# Authentication dependency
async def get_current_user(request: Request):
    # Get token from session
    token = request.session.get("access_token")

    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")    

    if not token:
        print("No token found in session or header")
        # Return None instead of raising an exception
        return None
    
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
    
# Error handling
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "404.html",
        {"request": request},
        status_code=404
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    status_messages = {
        404: "Not Found",
        500: "Internal Server Error",
    }
    
    if exc.status_code == 404:
        # We have specific templates for these
        template_name = f"{exc.status_code}.html"
    elif exc.status_code == 500:
        template_name = "500.html"
    else:
        # Use the general error template
        template_name = "error.html"
        
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "status_code": exc.status_code,
            "status_message": status_messages.get(exc.status_code, "Error"),
            "detail": exc.detail
        },
        status_code=exc.status_code
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return templates.TemplateResponse(
        "400.html",
        {"request": request},
        status_code=400
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse(
        "500.html",
        {"request": request},
        status_code=500
    )
    
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
    assignments = await fetch_all_assignments(token, user.userId)
    
    # Get completed evaluations for statistics
    completed_data = await fetch_all_completed_evaluations(token, user.userId)
    completed_count = len(completed_data)

    # Calculate average rating if there are completed evaluations
    avg_rating = 0
    unknown_count = 0

    if completed_count > 0:
        total_score = 0
        rating_count = 0
        
        for obj in completed_data:

            # Find this user's rating
            for rating in obj.get("ratings", []):
                if rating.get("userId") == user.userId:
                    # Check if this is an unknown object
                    if rating.get("metrics", {}).get("unknown_object", False):
                        unknown_count += 1
                        break  # Skip further processing for this object
                    
                    # Calculate average across metrics
                    metrics_to_count = ["accuracy", "completeness"]
                    for metric in metrics_to_count:
                        if metric in rating:
                            total_score += rating.get(metric, 0)
                            rating_count += 1
        
        if rating_count > 0:
            avg_rating = round(total_score / rating_count, 1)
        
        # Calculate unknown percentage
        unknown_percent = round((unknown_count / completed_count) * 100, 1) if completed_count > 0 else 0
    else:
        unknown_percent = 0
    
    # Get recent completed evaluations (limited to 3)
    recent_completed = []
    
    if completed_count > 0:
        # Sort by completion date (newest first) and take first 3
        sorted_completed = sorted(
            completed_data, 
            key=lambda x: next((a.get("completedAt", "") for a in x.get("assignments", []) 
                if a.get("userId") == user.userId), ""),
            reverse=True
        )
        recent_completed = sorted_completed[:3]
        
        # Add formatted user ratings to each object for display and fetch object details
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            
            for obj in recent_completed:
                user_metrics = {}
                obj["is_unknown_object"] = False
                        
                # Find this user's rating
                for rating in obj.get("ratings", []):
                    if rating.get("userId") == user.userId:
                        # Check if this was marked as an unknown object
                        if rating.get("metrics", {}).get("unknown_object", False):
                            obj["is_unknown_object"] = True
                            user_metrics = {"accuracy": 0, "completeness": 0, "clarity": 0}
                        else:
                            user_metrics = {
                                "accuracy": rating.get("accuracy", rating.get("score", 0)),
                                "completeness": rating.get("completeness", 0),
                                "clarity": rating.get("clarity", 0)
                            }
                        break
                
                # This line should be outside the inner for loop
                obj["ratings"] = user_metrics
                # Fetch object details
                object_response = await client.get(f"{API_URL}/api/objects/{obj.get('objectId')}", headers=headers)
                if object_response.status_code == 200:
                    obj["object"] = object_response.json().get("data", {})
                else:
                    obj["object"] = {"category": "Unknown"}  # Default fallback
    
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request, 
            "user": user, 
            "assignments": assignments,
            "pending_count": len(assignments),
            "completed_count": completed_count,
            "avg_rating": avg_rating,
            "unknown_count": unknown_count,
            "unknown_percent": unknown_percent,
            "recent_completed": recent_completed
        }
    )
# this is a redirect to the favicon to avoid error messages in the console
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return RedirectResponse('/static/favicon.ico')

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("login.html", {"request": request})

# Login route
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
                print(f"Auth service login error: {response.status_code} - {response.text}")
                return templates.TemplateResponse(
                    "login.html",
                    {"request": request, "error": f"Login failed: {response.status_code} - {response.text}"}
                )
            
            token_data = response.json()
            print(f"Got token with type: {type(token_data.get('access_token'))}")
            
            # Store token in session
            request.session["access_token"] = token_data["access_token"]
            print(f"Stored token in session: {request.session.get('access_token') is not None}")
            
            return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        print(f"Exception during login: {str(e)}")
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": str(e)}
        )

@app.get("/logout")
async def logout(request: Request):
    # Remove token from session
    request.session.pop("access_token", None)
    
    return RedirectResponse(url="/login")

# get the info for the object to evaluate
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

# big function to submit the evaluation
@app.post("/evaluate/{object_id}")
async def submit_evaluation(
    request: Request,
    object_id: str,
    accuracy: int = Form(None),
    completeness: int = Form(None),
    hallucinated: str = Form(None),
    unknown_object: str = Form(None),
    comments: str = Form(""),
    pageLoadTime: str = Form(None),  # Add this parameter
    user: Optional[User] = Depends(get_current_user)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Calculate time spent on page (in seconds)
    time_spent = None
    if pageLoadTime:
        try:
            # Convert to milliseconds and divide by 1000 to get seconds
            page_load_timestamp = int(pageLoadTime)
            current_time = int(time.time() * 1000)  # Current time in milliseconds
            time_spent = (current_time - page_load_timestamp) / 1000  # Time in seconds
            print(f"User spent {time_spent:.2f} seconds on evaluation page for object {object_id}")
        except ValueError:
            print(f"Invalid timestamp value: {pageLoadTime}")
    
    # Check if the user indicated they don't know what the object is
    if unknown_object == "true":
        # Submit evaluation with special flag for unknown object
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
            
            # Since the API requires score 1-5, use 1 but add metadata
            rating_data = {
                "score": 1,  # Use minimum score since API requires 1-5
                "metrics": {
                    "unknown_object": True,
                    "time_spent_seconds": time_spent  # Add time spent metric
                },
                "comment": "Evaluator indicated they do not know what this object is."
            }
            
            print(f"Submitting 'unknown object' evaluation for object {object_id}")
            
            response = await client.post(
                f"{API_URL}/api/objects/{object_id}/rate",
                json=rating_data,
                headers=headers
            )
            
            if response.status_code != 200 and response.status_code != 201:
                # Handle error case
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
                        "error": f"Failed to submit unknown object evaluation: {response.text}"
                    }
                )
        
        return RedirectResponse(url="/", status_code=303)
    
    # Normal evaluation flow
    if accuracy is None or completeness is None or hallucinated is None:
        # Handle missing required fields
        async with httpx.AsyncClient() as obj_client:
            headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
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
                "error": "Please fill out all required fields"
            }
        )
    
    # Submit normal evaluation
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        
        rating_data = {
            "score": accuracy,
            "metrics": {
                "accuracy": accuracy,
                "completeness": completeness,
                "hallucinated": hallucinated == "yes",
                "time_spent_seconds": time_spent  # Add time spent metric
            },
            "comment": comments if comments else None
        }

        print(f"Submitting evaluation for object {object_id} with data: {rating_data}")
        
        response = await client.post(
            f"{API_URL}/api/objects/{object_id}/rate",
            json=rating_data,
            headers=headers
        )
        
        # Handle response
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
                    "error": f"Failed to submit evaluation: {response.text}"
                }
            )
    
    return RedirectResponse(url="/", status_code=303)

# get the info for the object to review
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
        
        # Preprocess the object data to eliminate duplicate ratings
        if "ratings" in object_data and object_data["ratings"]:
            # Create a dictionary to store the most complete rating for each user
            user_ratings = {}
            
            for rating in object_data["ratings"]:
                user_id = rating.get("userId")
                if user_id not in user_ratings or len(rating.keys()) > len(user_ratings[user_id].keys()):
                    user_ratings[user_id] = rating
            
            # Replace ratings with deduplicated list
            object_data["ratings"] = list(user_ratings.values())
        
        # Process user rating for easier access in the template
        current_user_rating = None
        for rating in object_data.get("ratings", []):
            if rating.get("userId") == user.userId:
                current_user_rating = rating
                break
        
        # Find completion date if available
        completion_date = None
        for assignment in object_data.get("assignments", []):
            if assignment.get("userId") == user.userId and assignment.get("completedAt"):
                completion_date = assignment.get("completedAt")
                break
                
        # Prepare JSON data for the template
        user_json = json.dumps({
            "userId": user.userId,
            "email": user.email,
            "role": user.role
        })
        
        # Add processed data to the context
        context = {
            "request": request, 
            "user": user, 
            "object_json": json.dumps(object_data),
            "user_json": user_json,
            "user_rating_json": json.dumps(current_user_rating),
            "completion_date_json": json.dumps(completion_date)
        }
        
    return templates.TemplateResponse("review.html", context)

# Submit review route
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

# Assignments route for sidebar link
@app.get("/assignments", response_class=HTMLResponse)
async def assignments_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    assignments = await fetch_all_assignments(request.session.get("access_token"), user.userId)
    
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
    
    # Get all completed evaluations for the user
    token = request.session.get('access_token')
    evaluations = await fetch_all_completed_evaluations(token, user.userId)
    
    # Process evaluations to mark unknown objects
    for eval in evaluations:
        for rating in eval.get("ratings", []):
            if rating.get("userId") == user.userId:
                # Check if this was marked as an unknown object
                eval["is_unknown_object"] = rating.get("metrics", {}).get("unknown_object", False)
                break
    
    return templates.TemplateResponse(
        "completed.html",
        {"request": request, "user": user, "evaluations": evaluations}
    )

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
            json={"userId": user_id, "objectId": model_id},
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "user": user, "error": "Failed to assign object"}
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

# Upload CSV to assign models
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
            if "userId" not in row or "objectId" not in row:
                print("Missing required fields in row:", row)
                errors.append(f"Row {csv_reader.line_num}: Missing required fields (userId, objectId)")
                continue
                
            # Create assignment
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
                response = await client.post(
                    f"{API_URL}/api/assignments",
                    json={"userId": row["userId"], "objectId": row["objectId"]},
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

# Export users to CSV (used to help our assignment process)
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
   
    # Create CSV with additional fields for unknown object statistics
    output = io.StringIO()
    fieldnames = ["userId", "email", "role", "total_evaluations", "unknown_objects", "unknown_percent"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
   
    # For each user, fetch their evaluations to calculate unknown object stats
    for user_item in users_data:
        user_id = user_item.get("userId", "")
        
        # Get completed evaluations for this user
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
            response = await client.post(
                f"{API_URL}/api/completed",
                json={"userId": user_id},
                headers=headers
            )
            
            total_evaluations = 0
            unknown_count = 0
            unknown_percent = 0
            
            if response.status_code == 200:
                completed_data = response.json().get("data", [])
                total_evaluations = len(completed_data)
                
                # Count unknown objects
                for obj in completed_data:
                    for rating in obj.get("ratings", []):
                        if rating.get("userId") == user_id:
                            if rating.get("metrics", {}).get("unknown_object", False):
                                unknown_count += 1
                            break
                
                # Calculate percentage
                if total_evaluations > 0:
                    unknown_percent = round((unknown_count / total_evaluations) * 100, 1)
        
        # Write user data with stats to CSV
        writer.writerow({
            "userId": user_id,
            "email": user_item.get("email", ""),
            "role": user_item.get("role", ""),
            "total_evaluations": total_evaluations,
            "unknown_objects": unknown_count,
            "unknown_percent": f"{unknown_percent}%"
        })
   
    output.seek(0)
   
    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=users_with_stats.csv"
   
    return response

# Export all object IDs to CSV
@app.get("/admin/export_objects")
async def export_objects_csv(request: Request, user: Optional[User] = Depends(get_current_user)):
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
   
    # We'll need to paginate through all objects
    all_objects = []
    page = 1
    limit = 100  # Fetch more objects per request to reduce number of API calls
   
    # Get token from session
    token = request.session.get("access_token")
    if not token:
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "user": user, "error": "Authentication token not found"}
        )
   
    # Paginate through all objects
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
       
        while True:
            response = await client.get(f"{API_URL}/api/objects?page={page}&limit={limit}", headers=headers)
           
            if response.status_code != 200:
                return templates.TemplateResponse(
                    "admin.html",
                    {"request": request, "user": user, "error": f"Failed to fetch objects: {response.status_code}"}
                )
           
            data = response.json()
            objects = data.get("data", [])
            all_objects.extend(objects)
           
            # Check if we've reached the last page
            if page >= data.get("pages", 0) or len(objects) == 0:
                break
               
            page += 1
   
    # Create CSV with additional fields for unknown object statistics
    output = io.StringIO()
    fieldnames = [
        "objectId", 
        "description", 
        "category", 
        "averageRating", 
        "totalEvaluations", 
        "unknownCount", 
        "unknownPercent", 
        "createdAt"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
   
    for obj in all_objects:
        # Calculate unknown object stats
        total_evaluations = len(obj.get("ratings", []))
        unknown_count = 0
        hallucinated_count = 0
        total_time_spent = 0
        
        for rating in obj.get("ratings", []):
            if rating.get("metrics", {}).get("unknown_object", False):
                unknown_count += 1
            if rating.get("metrics", {}).get("hallucinated", False):
                hallucinated_count += 1
            if rating.get("metrics", {}).get("time_spent_seconds", None):
                total_time_spent += rating.get("metrics", {}).get("time_spent_seconds", 0)
        
        
        unknown_percent = 0
        hallucinated_percent = 0
        time_spent_avg = total_time_spent
        if total_evaluations > 0:
            unknown_percent = round((unknown_count / total_evaluations) * 100, 1)
            hallucinated_percent = round((hallucinated_count / total_evaluations) * 100, 1)

            time_spent_avg = round((total_time_spent / total_evaluations), 1)

        average_ratings = obj.get("averageRatings", {})
            
        writer.writerow({
            "objectId": obj.get("objectId", ""),
            "description": obj.get("description", "")[:100] + "..." if obj.get("description", "") else "",  # Truncate long descriptions
            "category": obj.get("category", ""),
            "averageAccuracy": average_ratings.get("accuracy", 0),
            "averageCompleteness": average_ratings.get("completeness", 0),
            "averageClarity": average_ratings.get("clarity", 0),
            "totalEvaluations": total_evaluations,
            "unknownCount": unknown_count,
            "unknownPercent": f"{unknown_percent}%",
            "hallucinatedCount": hallucinated_count,
            "hallucinatedPercent": f"{hallucinated_percent}%",
            "timeSpentAvg": time_spent_avg,
            "createdAt": obj.get("createdAt", "")
        })
   
    output.seek(0)
   
    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=objects_with_stats.csv"
   
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

@app.get("/admin/reviews", response_class=HTMLResponse)
async def admin_reviews_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    # Only allow admin users
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Get all users with their reviews
    users_with_reviews = []
    
    # First get all users
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        
        # Get users
        users_response = await client.get(f"{AUTH_URL}/users", headers=headers)
        if users_response.status_code != 200:
            return templates.TemplateResponse(
                "admin_reviews.html",
                {"request": request, "user": user, "users_with_reviews": [], "error": "Failed to fetch users"}
            )
        
        users = users_response.json().get("data", [])
        
        # For each user, get their completed reviews
        for user_item in users:
            # Get completed evaluations for this user
            completed_response = await client.post(
                f"{API_URL}/api/completed",
                json={"userId": user_item["userId"]},
                headers=headers
            )
            
            if completed_response.status_code == 200:
                completed_data = completed_response.json().get("data", [])
                
                # Add to our list if they have any reviews
                if completed_data:
                    users_with_reviews.append({
                        "user": user_item,
                        "reviews": completed_data,
                        "review_count": len(completed_data)
                    })
    
    return templates.TemplateResponse(
        "admin_reviews.html",
        {"request": request, "user": user, "users_with_reviews": users_with_reviews}
    )

@app.get("/admin/edit_review/{object_id}", response_class=HTMLResponse)
async def edit_review_page(request: Request, object_id: str, user: Optional[User] = Depends(get_current_user)):
    # Check if user is admin
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Get user_id from query parameter
    user_id = request.query_params.get("user_id")
    if not user_id:
        return RedirectResponse(url=f"/review/{object_id}", status_code=303)
    
    # Get object details
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        response = await client.get(f"{API_URL}/api/objects/{object_id}", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Object not found")
        
        object_data = response.json().get("data", {})
    
    # Find the specific rating for the given user
    user_rating = None
    for rating in object_data.get("ratings", []):
        if rating.get("userId") == user_id:
            user_rating = rating
            break
    
    if not user_rating:
        return RedirectResponse(url=f"/review/{object_id}", status_code=303)
    
    return templates.TemplateResponse(
        "edit_review.html",
        {
            "request": request, 
            "user": user, 
            "object": object_data, 
            "edit_user_id": user_id,
            "rating": user_rating
        }
    )

@app.post("/admin/update_review/{object_id}")
async def update_review(
    request: Request,
    object_id: str,
    user_id: str = Form(...),
    accuracy: int = Form(...),
    completeness: int = Form(...),
    hallucinated: str = Form(...),
    comment: str = Form(""),
    user: Optional[User] = Depends(get_current_user)
):
    # Check if user is admin
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
    
    # Prepare updated rating data
    rating_data = {
        "objectId": object_id,
        "userId": user_id,
        "rating": {
            "accuracy": accuracy,
            "completeness": completeness,
            "metrics": {
                "accuracy": accuracy,
                "completeness": completeness,
                "hallucinated": hallucinated == "yes"
            },
            "comment": comment if comment.strip() else None
        }
    }
    
    # Update rating in API
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
        
        response = await client.put(
            f"{API_URL}/api/ratings/{object_id}/{user_id}",
            json=rating_data,
            headers=headers
        )
        
        if response.status_code != 200 and response.status_code != 201:
            # If update fails, redirect to edit page with error
            return templates.TemplateResponse(
                "edit_review.html",
                {
                    "request": request, 
                    "user": user, 
                    "object_id": object_id,
                    "edit_user_id": user_id,
                    "error": f"Failed to update review: {response.text}"
                }
            )
    
    return RedirectResponse(url=f"/review/{object_id}", status_code=303)

@app.post("/admin/delete_review/{object_id}")
async def delete_review(
    request: Request,
    object_id: str,
    user_id: str = Form(...),
    user: Optional[User] = Depends(get_current_user)
):
    # Check if user is admin
    if not user or user.role != "admin":
        return RedirectResponse(url="/login")
   
    # Delete rating in API
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
       
        response = await client.delete(
            f"{API_URL}/api/ratings/{object_id}/{user_id}",
            headers=headers
        )
       
        if response.status_code != 200 and response.status_code != 204:
            # If deletion fails, redirect to review page with error
            return templates.TemplateResponse(
                "review.html",
                {
                    "request": request, 
                    "user": user, 
                    "object_id": object_id,
                    "error": f"Failed to delete review: {response.text}"
                }
            )
   
    # Redirect to admin panel after successful deletion
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info", workers=4)