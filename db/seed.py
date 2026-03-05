"""
ABC Real Estate - MongoDB Seed Script
Loads the training CSV into MongoDB so you can browse it in Compass.
Run: python db/seed.py
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME",   "abc_realestate")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db     = client[DB_NAME]

# Load CSV
df = pd.read_csv("data/rwanda_housing.csv")
records = df.to_dict(orient="records")

# Add a timestamp to each record
for r in records:
    r["seeded_at"] = datetime.now(timezone.utc)

# Insert into MongoDB
col = db["listings"]
col.drop()   # fresh start each time
result = col.insert_many(records)

print(f"Seeded {len(result.inserted_ids)} listings into MongoDB")
print(f"Database : {DB_NAME}")
print(f"Collection: listings")
print()
print("Open MongoDB Compass and connect to:")
print(f"  {MONGO_URI}")
print(f"  Database: {DB_NAME}")
print("  You will see 3 collections:")
print("   - listings     (all 2,000 training properties)")
print("   - predictions  (filled when you call /predict)")
print("   - validations  (filled when you call /validate-listing)")
