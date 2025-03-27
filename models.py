# models.py
from pydantic import BaseModel, EmailStr
from typing import Dict
from datetime import datetime
from typing import List

class AnalysisResult(BaseModel):
    email: EmailStr
    image: str
    analysis: Dict[str, str]

class Feedback(BaseModel):
    email: EmailStr
    message: str
    timestamp: datetime = datetime.utcnow()

class Skincare(BaseModel):
    gender: str
    condition: str
    tips: List[str]
    products: List[str]
    images: List[str]