import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
from jose import JWTError, jwt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Objaverse Research Portal")

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

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: Optional[User] = Depends(get_current_user)):
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
        "home.html",
        {"request": request, "user": user, "assignments": assignments}
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))