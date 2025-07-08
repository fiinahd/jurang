from motor.motor_asyncio import AsyncIOMotorClient # type: ignore
from .config import settings

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.DATABASE_NAME]