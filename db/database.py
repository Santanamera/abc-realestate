"""
ABC Real Estate - MongoDB Connection & Collection Helpers
"""
import os
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME",   "abc_realestate")

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")   # test connection
    db = client[DB_NAME]
    print(f"Connected to MongoDB  ->  {DB_NAME}")
except ConnectionFailure as e:
    print(f"WARNING: Could not connect to MongoDB: {e}")
    print("API will run but data will NOT be saved.")
    client = None
    db     = None

# ── Collections ───────────────────────────────────────────────────────────────
def get_predictions_col():
    """predictions  — every /predict call"""
    return db["predictions"] if db is not None else None

def get_validations_col():
    """validations  — every /validate-listing call"""
    return db["validations"] if db is not None else None

def get_listings_col():
    """listings     — house listings submitted to the platform"""
    return db["listings"] if db is not None else None

# ── Helpers ───────────────────────────────────────────────────────────────────
def now():
    return datetime.now(timezone.utc)

def save_prediction(house_data: dict, result: dict):
    col = get_predictions_col()
    if col is None:
        return None
    doc = {
        "timestamp":    now(),
        "input":        house_data,
        "predicted_price_usd":   result["predicted_price_usd"],
        "price_range_low_usd":   result["price_range_low_usd"],
        "price_range_high_usd":  result["price_range_high_usd"],
        "district":     house_data.get("district"),
        "property_type":house_data.get("property_type"),
        "flag":         result.get("flag"),
    }
    inserted = col.insert_one(doc)
    return str(inserted.inserted_id)

def save_validation(house_data: dict, listed_price: int, result: dict):
    col = get_validations_col()
    if col is None:
        return None
    doc = {
        "timestamp":           now(),
        "input":               house_data,
        "listed_price_usd":    listed_price,
        "predicted_price_usd": result["predicted_price_usd"],
        "price_range_low_usd": result["price_range_low_usd"],
        "price_range_high_usd":result["price_range_high_usd"],
        "deviation_pct":       result["deviation_pct"],
        "verdict":             result["verdict"],
        "district":            house_data.get("district"),
        "property_type":       house_data.get("property_type"),
    }
    inserted = col.insert_one(doc)
    return str(inserted.inserted_id)

def get_recent_predictions(limit: int = 20):
    col = get_predictions_col()
    if col is None:
        return []
    docs = col.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
    return list(docs)

def get_flagged_listings(limit: int = 50):
    col = get_validations_col()
    if col is None:
        return []
    docs = col.find(
        {"verdict": "FLAGGED"},
        {"_id": 0}
    ).sort("timestamp", DESCENDING).limit(limit)
    return list(docs)

def get_stats():
    """Summary stats for the dashboard."""
    preds = get_predictions_col()
    vals  = get_validations_col()
    if preds is None or vals is None:
        return {}
    return {
        "total_predictions":  preds.count_documents({}),
        "total_validations":  vals.count_documents({}),
        "flagged_listings":   vals.count_documents({"verdict": "FLAGGED"}),
        "fair_listings":      vals.count_documents({"verdict": "FAIR"}),
        "review_listings":    vals.count_documents({"verdict": "REVIEW"}),
        "by_district": list(preds.aggregate([
            {"$group": {"_id": "$district", "count": {"$sum": 1},
                        "avg_price": {"$avg": "$predicted_price_usd"}}},
            {"$sort":  {"count": -1}}
        ])),
    }
