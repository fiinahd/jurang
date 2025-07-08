from fastapi import APIRouter
from typing import Optional
from ..crud import get_labeled_data
from ..utils.split_train_eval import split_data
from ..utils.tfidf_knn import TfidfKNN
from ..schemas import TrainResponse
from sklearn.metrics import accuracy_score

router = APIRouter(prefix="/train")

@router.post("/", response_model=TrainResponse, tags=["train"] )
async def train_model(test_size: Optional[float] = 0.2):
    items = await get_labeled_data()
    texts = [it['aspect_text'] for it in items]
    labels = [it['label'] for it in items]
    train_texts, test_texts, train_labels, test_labels = split_data(texts, labels, test_size=test_size)
    model = TfidfKNN()
    model.fit(train_texts, train_labels)
    model.save('model_tfidf_knn.pkl')
    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    return {"accuracy": acc, "train_size": len(train_texts), "test_size": len(test_texts)}