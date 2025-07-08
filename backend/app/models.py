# from bson import ObjectId # type: ignore
from bson.objectid import ObjectId
from pydantic import BaseModel, Field
from typing import List, Optional

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

class RawReview(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    text: str

class CleanedReview(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    raw_id: PyObjectId
    cleaned_text: str
    tokens: List[str]

class AspectExtraction(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    cleaned_id: PyObjectId
    aspects: List[str]
    freqs: List[int]

class LabelItem(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    aspect_id: PyObjectId
    aspect_text: str
    label: Optional[str] = None  # "positive" | "negative" | "neutral"

class SentimentResult(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    aspect_id: PyObjectId
    sentiment: str