from pydantic import BaseModel, Field
from typing import List, Dict, Any

class AspectSelection(BaseModel):
    """Model untuk menerima daftar aspek dan persentase sampling."""
    aspects: List[str]
    sampling_percentage: int

class LabeledItem(BaseModel):
    """Model untuk satu baris data yang telah dilabeli."""
    id: int
    cleaned_review: str
    detected_aspects: str
    sentiment: str

class LabelingPayload(BaseModel):
    """Model untuk menerima seluruh data yang telah dilabeli."""
    labels: List[LabeledItem]

# --- MODEL BARU UNTUK VISUALISASI ---
class NetSentimentScore(BaseModel):
    aspect: str
    score: float

class SentimentDistribution(BaseModel):
    positif: int
    negatif: int
    netral: int

class WordCloudData(BaseModel):
    positif: List[List[Any]] = Field(default_factory=list)
    negatif: List[List[Any]] = Field(default_factory=list)
    netral: List[List[Any]] = Field(default_factory=list)

class AspectDetails(BaseModel):
    sentiment_distribution: SentimentDistribution
    word_clouds: WordCloudData

class VisualizationData(BaseModel):
    net_sentiment_scores: List[NetSentimentScore]
    aspect_details: Dict[str, AspectDetails]
    all_aspects: List[str]