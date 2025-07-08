from fastapi import APIRouter
from ..schemas import SelectRowsSchema
from ..crud import get_raw_preview

router = APIRouter(prefix="/select")

@router.get("/preview", tags=["select"])
async def preview(limit: int = 20):
    raws = await get_raw_preview(limit)
    return raws

@router.post("/", tags=["select"])
async def select_rows(sel: SelectRowsSchema):
    return {"selected_ids": sel.ids}