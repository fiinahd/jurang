# from bson import ObjectId
from bson.objectid import ObjectId
from .database import db

async def insert_raw(texts: list):
    docs = [{"text": t} for t in texts]
    res = await db.raw_reviews.insert_many(docs)
    return res.inserted_ids

async def get_raw_preview(limit=20):
    cursor = db.raw_reviews.find().limit(limit)
    return await cursor.to_list(length=limit)

async def insert_cleaned(raw_id: ObjectId, cleaned_text, tokens):
    doc = {"raw_id": raw_id, "cleaned_text": cleaned_text, "tokens": tokens}
    res = await db.cleaned_reviews.insert_one(doc)
    return res.inserted_id

async def insert_aspects(cleaned_id: ObjectId, aspects, freqs):
    doc = {"cleaned_id": cleaned_id, "aspects": aspects, "freqs": freqs}
    res = await db.extracted_aspects.insert_one(doc)
    return res.inserted_id

async def insert_label_item(aspect_id: ObjectId, aspect_text: str):
    await db.label_items.update_one(
        {"aspect_id": aspect_id, "aspect_text": aspect_text},
        {"$setOnInsert": {"aspect_id": aspect_id, "aspect_text": aspect_text, "label": None}},
        upsert=True
    )

async def get_unlabeled(limit=500):
    cursor = db.label_items.find({"label": None}).limit(limit)
    return await cursor.to_list(length=limit)

async def get_labeled_data():
    cursor = db.label_items.find({"label": {"$ne": None}})
    return await cursor.to_list(length=None)

async def update_label(item_id: ObjectId, label: str):
    await db.label_items.update_one({"_id": item_id}, {"$set": {"label": label}})

async def insert_sentiment(aspect_id: ObjectId, sentiment: str):
    doc = {"aspect_id": aspect_id, "sentiment": sentiment}
    res = await db.sentiment_results.insert_one(doc)
    return res.inserted_id