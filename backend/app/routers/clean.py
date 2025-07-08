from fastapi import APIRouter
# from bson import ObjectId
from bson.objectid import ObjectId
from ..database import db
from ..crud import insert_cleaned
from ..utils.preprocessing import clean_text

router = APIRouter(prefix="/clean")

@router.post("/", tags=["clean"] )
async def clean(selected_ids: list[str]):
    results = []
    for sid in selected_ids:
        text_doc = await db.raw_reviews.find_one({"_id": ObjectId(sid)})
        cleaned, tokens = clean_text(text_doc['text'])
        cid = await insert_cleaned(ObjectId(sid), cleaned, tokens)
        results.append({"cleaned_id": str(cid), "cleaned_text": cleaned})
    return results