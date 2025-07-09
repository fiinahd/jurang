from pydantic import BaseModel
from typing import List

class AspectSelection(BaseModel):
    """Model untuk menerima daftar aspek dan persentase sampling."""
    aspects: List[str]
    sampling_percentage: int # Nilai antara 1-100

class LabeledItem(BaseModel):
    """Model untuk satu baris data yang telah dilabeli."""
    id: int
    cleaned_review: str
    detected_aspects: str
    sentiment: str

class LabelingPayload(BaseModel):
    """Model untuk menerima seluruh data yang telah dilabeli."""
    labels: List[LabeledItem]