from fastapi import APIRouter
from typing import List
# from bson import ObjectId
from bson.objectid import ObjectId
from ..crud import insert_label_item, get_unlabeled, update_label
from ..schemas import LabelInitItem, LabelSchema

router = APIRouter(prefix="/label")

@router.post("/init", tags=["label"] )
async def init_label(items: List[LabelInitItem]):
    for it in items:
        await insert_label_item(ObjectId(it.aspect_id), it.aspect_text)
    return {"initialized": len(items)}

@router.get("/", tags=["label"] )
async def get_to_label(limit: int = 500):
    items = await get_unlabeled(limit)
    return items

@router.post("/", tags=["label"] )
async def post_label(batch: List[LabelSchema]):
    for it in batch:
        await update_label(ObjectId(it.id), it.label)
    return {"updated": len(batch)}