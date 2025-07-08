from fastapi import APIRouter
# from bson import ObjectId
from bson.objectid import ObjectId
from ..database import db
from ..crud import insert_aspects
from ..utils.postagging import pos_tags
from ..utils.rulebasedextract import extract_nouns

router = APIRouter(prefix="/extract")

@router.post("/", tags=["extract"] )
async def extract(cleaned_ids: list[str]):
    out = []
    for cid in cleaned_ids:
        doc = await db.cleaned_reviews.find_one({"_id": ObjectId(cid)})
        tagged = pos_tags(doc['tokens'])
        aspects, freqs = extract_nouns(tagged)
        aid = await insert_aspects(ObjectId(cid), aspects, freqs)
        out.append({"aspect_id": str(aid), "aspects": aspects})
    return out