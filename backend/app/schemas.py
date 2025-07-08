from pydantic import BaseModel
from typing import List

class UploadSchema(BaseModel):
    texts: List[str]

class SelectRowsSchema(BaseModel):
    ids: List[str]

class LabelInitItem(BaseModel):
    aspect_id: str
    aspect_text: str

class LabelSchema(BaseModel):
    id: str
    label: str

class TrainResponse(BaseModel):
    accuracy: float
    train_size: int
    test_size: int

class PredictRequest(BaseModel):
    aspect_id: str
    aspect_text: str