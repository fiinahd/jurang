from fastapi import APIRouter
from ..schemas import UploadSchema
from ..crud import insert_raw

router = APIRouter(prefix="/upload")

@router.post("/", tags=["upload"] )
async def upload(data: UploadSchema):
    ids = await insert_raw(data.texts)
    return {"job_id": [str(i) for i in ids]}