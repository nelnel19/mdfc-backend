# database.py
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Collections
users_collection = db.users
medication_collection = db.medication
results_collection = db.results
feedback_collection = db.feedback  # ✅ Added feedback collection
medications_collection = db.medications
skincare_collection = db.skincare
routine_collection = db.routine
