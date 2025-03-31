from fastapi import FastAPI, HTTPException, Depends, Request, Body, Query
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
import secrets
import aiosmtplib
from email.mime.text import MIMEText
from starlette.middleware.cors import CORSMiddleware  
from typing import List, Optional, Union
from bson import ObjectId
from fastapi import Form, File, UploadFile
from cloudinary_config import upload_images
from database import medication_collection
import jwt
from datetime import datetime, timedelta
from models import AnalysisResult, Feedback, Skincare
from database import db
from database import results_collection, feedback_collection, medications_collection, skincare_collection, routine_collection
from typing import Optional
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
from fastapi import BackgroundTasks
import asyncio
import smtplib
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from dotenv import load_dotenv
import requests
import shutil
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow only the React frontend's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allowed HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Mailtrap Credentials
MAILTRAP_USERNAME = os.getenv("MAILTRAP_USERNAME")
MAILTRAP_PASSWORD = os.getenv("MAILTRAP_PASSWORD")
MAILTRAP_HOST = os.getenv("MAILTRAP_HOST")
MAILTRAP_PORT = int(os.getenv("MAILTRAP_PORT"))

#login token
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token expires in 1 hour

# Temporary token storage (Use Redis/Database in production)
reset_tokens = {}
class SkincareRecommendation(BaseModel):
    email: str  
    gender: str
    skin_type: str
    conditions: List[str]
    recommendations: List[dict]  
    images: list[str]  # URLs of images from Cloudinary
    
class MedicationSchema(BaseModel):
    disease: str
    description: Optional[str] = "No description available"
    tips: List[str]
    products: List[str]
    images: List[str]

class SaveMedicationRequest(BaseModel):
    email: str
    disease: str
    tips: list
    products: list
    images: list

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    age: int
    gender: str
    role: str = "user"  # Default role is "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    token: str
    new_password: str

class AnalysisResult(BaseModel):
    analysis: dict
    email: str

async def send_deactivation_email(email: str):
    """Send an email notification to the user about account deactivation."""
    subject = "Account Deactivation Notification"
    body = """
    Hello,

    Your account will be deactivated for 15 hours due to incorrect usage of our system. 
    If you believe this is a mistake, please contact our support team.

    Thank you for your understanding.

    Best regards,
    Your System Admin
    """

    # Create the email message
    message = MIMEText(body)
    message["From"] = "MEdiFaceCare@gmail.com"  # Replace with your sender email
    message["To"] = email
    message["Subject"] = subject

    try:
        # Connect to Mailtrap SMTP server
        smtp = aiosmtplib.SMTP(hostname=MAILTRAP_HOST, port=MAILTRAP_PORT)
        await smtp.connect()
        # No need for starttls() as per your setup
        await smtp.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
        await smtp.send_message(message)
        await smtp.quit()
        print(f"Deactivation email sent to {email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

async def check_and_reactivate_users():
    """Check for users who have been deactivated for more than 24 hours and reactivate them."""
    current_time = datetime.utcnow()
    deactivated_users = await users_collection.find(
        {"status": "deactivated", "deactivated_at": {"$lt": current_time - timedelta(hours=24)}}
    ).to_list(length=None)

    for user in deactivated_users:
        await users_collection.update_one(
            {"_id": user["_id"]}, 
            {"$set": {"status": "active"}, "$unset": {"deactivated_at": ""}}
        )
        print(f"User {user['email']} has been reactivated.")

@app.on_event("startup")
async def startup_event():
    """Schedule the background task to run periodically."""
    import asyncio
    asyncio.create_task(periodic_reactivation_check())

async def periodic_reactivation_check():
    """Periodically check and reactivate users."""
    while True:
        await check_and_reactivate_users()
        await asyncio.sleep(3600)  # Check every hour

async def get_user(email: str):
    return await users_collection.find_one({"email": email})

async def send_thank_you_email(email: str):
    # Create the email message
    subject = "Thank You for Signing Up on MediFaceCare"
    body = """
    Dear User,

    Thank you for signing up on MediFaceCare! We are thrilled to have you as part of our community.

    At MediFaceCare, we are committed to providing you with the best experience in managing your health and wellness. Whether you're here to book appointments, consult with healthcare professionals, or explore our services, we're here to support you every step of the way.

    If you have any questions or need assistance, please don't hesitate to reach out to our support team at support@medifacecare.com.

    Once again, welcome to MediFaceCare. We look forward to serving you!

    Best regards,
    The MediFaceCare Team
    """

    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = "MediFaceCare@example.com"  # Use a valid sender email
    msg['To'] = email
    msg['Subject'] = subject

    # Attach the body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Connect to Mailtrap and send the email
    try:
        print(f"Attempting to connect to Mailtrap: {MAILTRAP_HOST}:{MAILTRAP_PORT}")
        with smtplib.SMTP(MAILTRAP_HOST, MAILTRAP_PORT) as server:
            print("Starting TLS...")
            server.starttls()
            print(f"Logging in with username: {MAILTRAP_USERNAME}")
            server.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
            print(f"Sending email to {email}...")
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        print(f"Thank you email sent to {email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Registration endpoint
@app.post("/register")
async def register(user: UserCreate):
    try:
        # Check if the user already exists
        existing_user = await get_user(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash the password
        hashed_password = pwd_context.hash(user.password)

        # Prepare user data for insertion
        user_data = user.dict()
        user_data["password"] = hashed_password
        user_data["status"] = "active"  # Set default status to active

        # Insert the user into the database
        await users_collection.insert_one(user_data)  # Use await here

        # Send a thank-you email
        try:
            await send_thank_you_email(user.email)  # Await the email sending function
        except Exception as e:
            print(f"Failed to send email: {e}")
            # Log the error but do not fail the registration

        return {"message": "User registered successfully"}
    except Exception as e:
        print(f"Error during registration: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Function to generate JWT token
def create_jwt_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/login")
async def login(user: UserLogin):
    existing_user = await get_user(user.email)
    
    # Check if the user exists and the password is correct
    if not existing_user or not pwd_context.verify(user.password, existing_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Check if the user is archived or deactivated
    if existing_user.get("status") in ["archived", "deactivated"]:
        raise HTTPException(status_code=403, detail="Your account is archived or deactivated. Please contact support.")
    
    # Create JWT token
    token_data = {"sub": user.email}
    access_token = create_jwt_token(token_data)

    # Update last login time
    last_login_time = datetime.now(timezone.utc)
    await users_collection.update_one(
        {"email": user.email}, 
        {"$set": {"last_login": last_login_time}}
    )

    # Return user data including role
    return {
        "email": user.email,
        "role": existing_user.get("role", "user"),  # Default to "user" if role is not set
        "access_token": access_token
    }

# ======== Forgot Password Functionality ========

async def send_reset_email(email: str, token: str):
    """Send password reset email via Mailtrap"""
    reset_link = f"http://localhost:8000/reset-password?email={email}&token={token}"
    message = MIMEText(f"Click the link to reset your password: {reset_link}")
    message["From"] = "noreply@example.com"
    message["To"] = email
    message["Subject"] = "Password Reset Request"

    try:
        smtp = aiosmtplib.SMTP(hostname=MAILTRAP_HOST, port=MAILTRAP_PORT)
        await smtp.connect()
        # REMOVE THIS LINE -> await smtp.starttls()  
        await smtp.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
        await smtp.send_message(message)
        await smtp.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Handle forgot password request"""
    user = await users_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")
    
    token = secrets.token_hex(16)
    reset_tokens[request.email] = token  # Store the token temporarily
    
    email_sent = await send_reset_email(request.email, token)
    if not email_sent:
        raise HTTPException(status_code=500, detail="Failed to send email")

    return {"message": "Password reset email sent"}

@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Handle password reset"""
    stored_token = reset_tokens.get(request.email)
    if not stored_token or stored_token != request.token:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    hashed_password = pwd_context.hash(request.new_password)
    await users_collection.update_one({"email": request.email}, {"$set": {"password": hashed_password}})
    
    del reset_tokens[request.email]  # Remove the token after use
    return {"message": "Password reset successful"}

#users
def user_serializer(user):
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "username": user["username"],
        "age": user["age"],
        "gender": user["gender"],
        "profile_image": user.get("profile_image", None),
        "status": user["status"],
        "last_login": user.get("last_login", None),
        "deactivated_at": user.get("deactivated_at", None),  # Include deactivation timestamp
        "role": user.get("role", "user")  # Include role field with a default value of "user"
    }

# ✅ Get all users route
@app.get("/users")
async def get_all_users():
    try:
        users = await users_collection.find().to_list(length=None)
        if not users:
            raise HTTPException(status_code=404, detail="No users found")

        return {"users": [user_serializer(user) for user in users]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")

@app.get("/users/{email}")
async def get_user_by_email(email: str):
    """Fetch user details based on email, including last login time."""
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_serializer(user)



    #medication crud
# Insert Medication Data with Images
@app.post("/medications/")
async def add_medication(
    disease: str = Form(...),
    description: str = Form(...),
    tips: List[str] = Form(...),
    products: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    image_urls = await upload_images(files)
    
    medication_data = {
        "disease": disease,
        "description": description,
        "tips": tips,
        "products": products,
        "images": image_urls
    }
    
    result = await medication_collection.insert_one(medication_data)
    return {"message": "Medication added successfully", "id": str(result.inserted_id)}

# Get All Medications
@app.get("/medications/", response_model=List[MedicationSchema])
async def get_medications():
    medications = await medication_collection.find().to_list(100)
    return medications

@app.get("/medications/{disease}", response_model=List[MedicationSchema])
async def get_medications(disease: str):
    """Fetch medications from MongoDB based on the predicted disease."""
    medications = await medication_collection.find({"disease": disease}).to_list(100)
    if not medications:
        raise HTTPException(status_code=404, detail="No medication found")  # Corrected indentation
    return medications

# Get Single Medication by ID
@app.get("/medications/{medication_id}", response_model=MedicationSchema)
async def get_medication(medication_id: str):
    medication = await medication_collection.find_one({"_id": ObjectId(medication_id)})
    if medication:
        return medication
    return {"error": "Medication not found"}

# Update Medication by ID
@app.put("/medications/{medication_id}")
async def update_medication(medication_id: str, updated_data: MedicationSchema):
    await medication_collection.update_one(
        {"_id": medication_id}, {"$set": updated_data.dict()}
    )
    return {"message": "Medication updated successfully"}

# Delete Medication by ID
@app.delete("/medications/{medication_id}")
async def delete_medication(medication_id: str):
    result = await medication_collection.delete_one({"_id": medication_id})
    if result.deleted_count == 1:
        return {"message": "Medication deleted successfully"}
    return {"error": "Medication not found"}

@app.post("/save-result")
async def save_result(
    image: UploadFile = File(...),  
    analysis: str = Form(...),
    email: str = Form(...)
):
    try:
        # Upload image to Cloudinary
        image_urls = await upload_images([image])
        image_url = image_urls[0] if image_urls else None

        if image_url is None:
            raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary")

        # Save analysis result in MongoDB with a timestamp
        result_dict = {
            "image": image_url,
            "analysis": eval(analysis),  # Convert string back to dictionary
            "email": email,
            "date_saved": datetime.utcnow()  # Add timestamp in UTC
        }
        await results_collection.insert_one(result_dict)
        
        return {"message": "Result saved successfully with image in Cloudinary"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save result: {str(e)}")

# Get All Analysis Results
from datetime import datetime, timezone

def result_serializer(result):
    """Serialize MongoDB result document into a JSON-friendly format."""
    date_saved = result.get("date_saved", None)  # Use the new date field
    
    # Ensure the date_saved field is in UTC and includes time zone information
    if isinstance(date_saved, datetime):
        if date_saved.tzinfo is None:
            date_saved = date_saved.replace(tzinfo=timezone.utc)  # Assume UTC if no time zone is specified
        date_saved_iso = date_saved.isoformat()  # This will include the time zone
    else:
        date_saved_iso = ""  # Fallback if date_saved is not a datetime object
    
    return {
        "id": str(result["_id"]),
        "email": result["email"],
        "image": result.get("image", ""),  # Ensure image key exists
        "analysis": result.get("analysis", {}),
        "date_saved": date_saved_iso  # Use the ISO-formatted date with time zone
    }

@app.get("/results")
async def get_all_results():
    """Fetch all analysis results from the database."""
    try:
        results = await results_collection.find().to_list(100)  # Limit to 100 results
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        return {"results": [result_serializer(result) for result in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

#feedback
@app.post("/feedback/")
async def submit_feedback(feedback: Feedback):
    try:
        feedback_dict = feedback.dict()
        result = await feedback_collection.insert_one(feedback_dict)
        return {"message": "Feedback submitted successfully", "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serializer for feedback
def feedback_serializer(feedback):
    return {
        "id": str(feedback["_id"]),
        "email": feedback["email"],
        "message": feedback["message"],
        "reply": feedback.get("reply", None),  # Include reply field, default to None if missing
        "timestamp": feedback.get("timestamp", None)  # Include timestamp
    }

# ✅ Get all feedback route
@app.get("/feedbacks")
async def get_all_feedbacks():
    try:
        feedbacks = await feedback_collection.find().to_list(length=None)  # Fetch all feedback
        if not feedbacks:
            raise HTTPException(status_code=404, detail="No feedback found")

        return {"feedbacks": [feedback_serializer(item) for item in feedbacks]}  # Format using serializer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feedback: {str(e)}")


class ReplyRequest(BaseModel):
    reply: str

@app.post("/feedback/{feedback_id}/reply")
async def reply_to_feedback(feedback_id: str, request: ReplyRequest):
    try:
        print(f"Feedback ID: {feedback_id}")
        print(f"Reply Payload: {request}")

        result = await feedback_collection.update_one(
            {"_id": ObjectId(feedback_id)},
            {"$set": {"reply": request.reply}}
        )
        if result.modified_count:
            return {"message": "Reply added successfully"}
        else:
            raise HTTPException(status_code=404, detail="Feedback not found")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/users/{email}")
async def update_user(
    email: str,
    username: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Update user details and profile image."""

    email = email.strip()  # Trim spaces or newlines

    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}

    if username is not None:
        update_data["username"] = username
    if age is not None:
        update_data["age"] = age
    if gender is not None:
        update_data["gender"] = gender

    if file:
        try:
            image_urls = await upload_images([file])
            if image_urls:
                update_data["profile_image"] = image_urls[0]
        except Exception as e:
            print(f"Image upload failed: {e}")

    if update_data:
        await users_collection.update_one({"email": email}, {"$set": update_data})

    return {"message": "User updated successfully"}

@app.put("/users/{email}")
async def update_user(
    email: str,
    username: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Update user details and profile image."""

    email = email.strip()  # Trim spaces or newlines

    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}

    if username is not None:
        update_data["username"] = username
    if age is not None:
        update_data["age"] = age
    if gender is not None:
        update_data["gender"] = gender

    if file:
        try:
            image_urls = await upload_images([file])
            if image_urls:
                update_data["profile_image"] = image_urls[0]
        except Exception as e:
            print(f"Image upload failed: {e}")

    if update_data:
        await users_collection.update_one({"email": email}, {"$set": update_data})

    return {"message": "User updated successfully"}
    

@app.put("/users/{email}/unarchive")
async def unarchive_user(email: str):
    """Unarchive a user by setting their status back to 'active'."""
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.get("status") != "archived":
        raise HTTPException(status_code=400, detail="User is not archived")

    await users_collection.update_one({"email": email}, {"$set": {"status": "active"}})

    return {"message": "User unarchived successfully"}


@app.put("/users/{email}/archive")
async def archive_user(email: str):
    """Archive a user by setting their status to 'archived'."""
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.get("status") == "archived":
        raise HTTPException(status_code=400, detail="User is already archived")

    await users_collection.update_one({"email": email}, {"$set": {"status": "archived"}})

    return {"message": "User archived successfully"}

@app.put("/users/{email}/deactivate")
async def deactivate_user(email: str):
    """Deactivate a user by setting their status to 'deactivated' and send a notification email."""
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.get("status") == "deactivated":
        raise HTTPException(status_code=400, detail="User is already deactivated")

    # Update the user's status to "deactivated" and store the deactivation timestamp
    deactivation_time = datetime.utcnow()
    await users_collection.update_one(
        {"email": email}, 
        {"$set": {"status": "deactivated", "deactivated_at": deactivation_time}}
    )

    # Send a deactivation email to the user
    email_sent = await send_deactivation_email(email)
    if not email_sent:
        raise HTTPException(status_code=500, detail="Failed to send deactivation email")

    return {"message": "User deactivated successfully. Notification email sent."}


@app.put("/users/{email}/reactivate")
async def reactivate_user(email: str):
    """Reactivate a user by setting their status back to 'active'."""
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.get("status") != "deactivated":
        raise HTTPException(status_code=400, detail="User is not deactivated")

    await users_collection.update_one(
        {"email": email}, 
        {"$set": {"status": "active"}, "$unset": {"deactivated_at": ""}}
    )

    return {"message": "User reactivated successfully"}   


    
@app.post("/save_medication/")
async def save_medication(data: SaveMedicationRequest):
    """Save medication recommendations under the user's email with timestamp."""
    try:
        saved_medication = {
            "email": data.email,
            "disease": data.disease,
            "tips": data.tips,
            "products": data.products,
            "images": data.images,
            "created_at": datetime.utcnow()  # Add timestamp in UTC format
        }
        
        result = await medications_collection.insert_one(saved_medication)
        if result.inserted_id:
            return {"message": "Medication saved successfully", "timestamp": saved_medication["created_at"]}
        else:
            raise HTTPException(status_code=500, detail="Failed to save medication")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving medication: {str(e)}")

@app.get("/get_medications/")
async def get_medications(start_date: str = Query(None, description="Start date in YYYY-MM-DD format"), 
                          end_date: str = Query(None, description="End date in YYYY-MM-DD format")):
    """Retrieve all saved medications within a date range."""
    try:
        query = {}
        if start_date and end_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            query["created_at"] = {"$gte": start_date, "$lte": end_date}

        medications = await medications_collection.find(query).to_list(length=100)

        if not medications:
            raise HTTPException(status_code=404, detail="No medications found")

        # Convert ObjectId and datetime to string for JSON response
        for med in medications:
            med["_id"] = str(med["_id"])
            med["created_at"] = med["created_at"].isoformat()
        
        return {"medications": medications}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving medications: {str(e)}")

@app.get("/user_medications/{email}", response_model=List[MedicationSchema])
async def get_user_medications(email: str):
    try:
        # ✅ Fetch medications for the user
        user_medications = await medications_collection.find({"email": email}).to_list(length=100)
        
        if not user_medications:
            return []  # Return an empty list if no medications are found

        # ✅ Deserialize MongoDB documents into Pydantic models
        return [MedicationSchema(**med) for med in user_medications]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

@app.get("/medications/", response_model=List[MedicationSchema])
async def get_all_medications():
    """Retrieve all saved medications."""
    try:
        medications = await medications_collection.find().to_list(length=1000)
        return [MedicationSchema(**med) for med in medications]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching medications: {str(e)}")


@app.post("/skincare/")
async def add_skincare(
    gender: str = Form(...),
    condition: str = Form(...),
    tips: List[str] = Form(...),
    products: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    """Add a new skincare recommendation with at least three tips and three images"""
    if len(tips) < 3:
        raise HTTPException(status_code=400, detail="Please provide at least three skincare tips")
    
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="Please upload at least three images")

    # Upload images to Cloudinary
    image_urls = await upload_images(files)
    
    skincare_data = {
        "gender": gender,
        "condition": condition,
        "tips": tips,
        "products": products,
        "images": image_urls
    }

    # Save skincare data to MongoDB
    result = await skincare_collection.insert_one(skincare_data)
    
    return {"message": "Skincare recommendation added successfully", "id": str(result.inserted_id)}


@app.get("/skincares/")
async def get_all_skincare():
    """Retrieve all skincare recommendations from the database"""
    skincare_list = await skincare_collection.find().to_list(length=None)
    
    if not skincare_list:
        return JSONResponse(content={"message": "No skincare recommendations found"}, status_code=404)
    
    # Convert ObjectId to string
    for skincare in skincare_list:
        skincare["_id"] = str(skincare["_id"])

    return {"skincare_recommendations": skincare_list}

@app.post("/save_skincare/")
async def save_skincare_routine(data: SkincareRecommendation):
    """Save skincare routine recommendations under the user's email in the 'Routine' collection."""
    try:
        # Prepare the data to save
        saved_routine = {
            "email": data.email,
            "gender": data.gender,
            "skin_type": data.skin_type,
            "conditions": data.conditions,
            "recommendations": data.recommendations,
            "images": data.images,  
            "created_at": datetime.utcnow()  # Add timestamp
        }

        # Save the data into MongoDB under the "Routine" collection
        result = await routine_collection.insert_one(saved_routine)
        if result.inserted_id:
            return {"message": "Skincare routine saved successfully!", "timestamp": saved_routine["created_at"]}
        else:
            raise HTTPException(status_code=500, detail="Failed to save skincare routine")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving skincare routine: {str(e)}")

@app.get("/get_skincare_routines/")
async def get_skincare_routines():
    """Retrieve all saved skincare routines."""
    try:
        routines = await routine_collection.find().to_list(length=100)

        if not routines:
            raise HTTPException(status_code=404, detail="No skincare routines found")

        # Convert ObjectId and handle missing 'created_at' field
        for routine in routines:
            routine["_id"] = str(routine["_id"])
            routine["created_at"] = routine.get("created_at", datetime.utcnow()).isoformat()  

        return {"routines": routines}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving skincare routines: {str(e)}")



@app.get("/get_skincare/{email}")
async def get_skincare_routines(email: str):
    """Get all skincare routine recommendations for a specific email."""
    try:
        # Fetch all routines matching the email
        routines = await routine_collection.find({"email": email}).to_list(length=None)

        if not routines:
            raise HTTPException(status_code=404, detail="No skincare routines found for this email")

        # Convert ObjectId to string and return the list of routines
        for routine in routines:
            routine["_id"] = str(routine["_id"])

        return routines  # Return a list of routines

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching skincare routines: {str(e)}")

# Face++ API Credentials
FACE_API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
API_KEY = "twZOWirK5l8ymNhccVJF1uetKAzQvEHx"
API_SECRET = "OuRkSXNxDtW4BNKP-QiHzKRqPaD7ET7a"


FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Define skin tone categories
SKIN_TONES = [
    ("Very Fair", (220, 200, 190)),
    ("Fair", (200, 180, 160)),
    ("Light-Medium", (180, 160, 140)),
    ("Medium", (160, 140, 120)),
    ("Tan", (140, 120, 100)),
    ("Deep Tan", (120, 100, 80)),
    ("Deep", (100, 80, 60)),
    ("Very Deep", (80, 60, 40)),
]


# Undertone classification reference
UNDERTONES = {
    "Cool": [(200, 180, 220), (180, 160, 200), (160, 140, 180)],
    "Warm": [(220, 200, 160), (200, 180, 140), (180, 160, 120)],
    "Neutral": [(190, 170, 150), (170, 150, 130), (150, 130, 110)]
}


def preprocess_image(image_bytes):
    """Preprocess image for model prediction."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image


def classify_skin_tone(avg_color):
    """Determine closest skin tone match."""
    min_diff = float("inf")
    best_match = "Unknown"
    for tone, ref_color in SKIN_TONES:
        diff = np.linalg.norm(np.array(avg_color) - np.array(ref_color))
        if diff < min_diff:
            min_diff = diff
            best_match = tone
    return best_match


def classify_undertone(avg_lab_color):
    """Determine undertone based on closest RGB match."""
    avg_rgb_color = cv2.cvtColor(
        np.uint8([[avg_lab_color]]), cv2.COLOR_LAB2RGB
    )[0][0]

    best_match = "Neutral"
    min_diff = float("inf")

    for undertone, colors in UNDERTONES.items():
        for ref_color in colors:
            diff = np.linalg.norm(np.array(avg_rgb_color) - np.array(ref_color))
            if diff < min_diff:
                min_diff = diff
                best_match = undertone

    return best_match if min_diff < 60 else "Neutral"  # Adjust threshold


@app.post("/analyze/")
async def analyze_skin(file: UploadFile = File(...)):
    """Analyze skin tone and undertone from an uploaded image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_image = np.array(image)

    # Convert image to LAB color space for better skin tone extraction
    lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    avg_color = np.mean(lab_image.reshape(-1, 3), axis=0)

    # Convert LAB avg color to RGB before classification
    skin_tone = classify_skin_tone(avg_color)
    undertone = classify_undertone(avg_color)  # Now uses correct color format

    return {"skin_tone": skin_tone, "undertone": undertone}


#face++
def analyze_face(image_path):
    """Send image to Face++ API and analyze face attributes."""
    with open(image_path, "rb") as image_file:
        files = {"image_file": image_file}
        data = {
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "return_attributes": "gender,skinstatus"
        }
        response = requests.post(FACE_API_URL, data=data, files=files)
    
    if response.status_code == 200:
        return response.json()
    return None

@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """Handles image upload and calls Face++ for analysis."""
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Analyze the image
    face_data = analyze_face(file_path)

    # Remove temp file
    os.remove(file_path)

    if not face_data or "faces" not in face_data or not face_data["faces"]:
        return {"error": "No face detected"}

    attributes = face_data["faces"][0]["attributes"]
    gender = attributes["gender"]["value"]
    skin_status = attributes.get("skinstatus", {})

    acne = skin_status.get("acne", "Unknown")
    dark_circles = skin_status.get("dark_circle", "Unknown")
    stain = skin_status.get("stain", "Unknown")

    return {
        "gender": gender,
        "acne": acne,
        "dark_circles": dark_circles,
        "stain": stain
    }



