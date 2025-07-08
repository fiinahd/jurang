from fastapi import APIRouter
from typing import List
# from bson import ObjectId
from bson.objectid import ObjectId
from ..crud import insert_sentiment
from ..utils.tfidf_knn import TfidfKNN
from ..schemas import PredictRequest

router = APIRouter(prefix="/sentiment")

@router.post("/", tags=["sentiment"] )
async def sentiment(items: List[PredictRequest]):
    model = TfidfKNN.load('model_tfidf_knn.pkl')
    texts = [it.aspect_text for it in items]
    preds = model.predict(texts)
    out = []
    for it, p in zip(items, preds):
        sid = await insert_sentiment(ObjectId(it.aspect_id), p)
        out.append({"aspect_id": it.aspect_id, "sentiment": p})
    return out