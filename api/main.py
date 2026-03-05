"""
ABC Real Estate - House Price Prediction API
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import joblib, json
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(
    title="ABC Real Estate - Price Prediction API",
    description="Fair market price prediction for Rwanda properties.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).parent.parent
model    = joblib.load(BASE / "model" / "house_price_model.pkl")
metadata = json.loads((BASE / "model" / "model_metadata.json").read_text())
print("Model loaded successfully")

MARGIN = 0.10

class HouseFeatures(BaseModel):
    district:           Literal["Gasabo","Kicukiro","Nyarugenge","Musanze","Rubavu","Huye"]
    property_type:      Literal["Apartment","Standalone Villa","Townhouse","Bungalow"]
    road_condition:     Literal["Tarmac","Gravel","Dirt"]
    bedrooms:           int   = Field(..., ge=1, le=10, example=3)
    bathrooms:          int   = Field(..., ge=1, le=10, example=2)
    size_sqm:           int   = Field(..., ge=20, le=1000, example=120)
    year_built:         int   = Field(..., ge=1960, le=2024, example=2015)
    has_garage:         bool  = Field(False)
    has_garden:         bool  = Field(False)
    has_swimming_pool:  bool  = Field(False)
    has_security:       bool  = Field(True)
    is_gated_community: bool  = Field(False)
    distance_to_cbd_km: float = Field(..., ge=0.1, le=60.0, example=5.5)

def to_df(h: HouseFeatures):
    return pd.DataFrame([{
        "district": h.district, "property_type": h.property_type,
        "road_condition": h.road_condition, "bedrooms": h.bedrooms,
        "bathrooms": h.bathrooms, "size_sqm": h.size_sqm,
        "year_built": h.year_built, "has_garage": int(h.has_garage),
        "has_garden": int(h.has_garden), "has_swimming_pool": int(h.has_swimming_pool),
        "has_security": int(h.has_security), "is_gated_community": int(h.is_gated_community),
        "distance_to_cbd_km": h.distance_to_cbd_km,
    }])

def predict(h: HouseFeatures, listed: Optional[int] = None):
    pred = float(model.predict(to_df(h))[0])
    low, high, mid = round(pred*(1-MARGIN),-2), round(pred*(1+MARGIN),-2), round(pred,-2)
    flag = None
    if listed:
        dev = abs(listed - pred) / pred
        if dev > 0.25:
            d = "above" if listed > pred else "below"
            flag = f"Listed ${listed:,} is {dev*100:.0f}% {d} fair value (${int(mid):,})"
    return {"predicted_price_usd":int(mid),"price_range_low_usd":int(low),
            "price_range_high_usd":int(high),"district":h.district,
            "property_type":h.property_type,"flag":flag}

@app.get("/")
def root():
    return {"service":"ABC Real Estate API","status":"running","model_r2":metadata["r2_test"]}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict")
def predict_price(house: HouseFeatures, listed_price: Optional[int] = None):
    try:
        return predict(house, listed_price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-listing")
def validate_listing(house: HouseFeatures, listed_price_usd: int):
    r   = predict(house)
    mid = r["predicted_price_usd"]
    dev = (listed_price_usd - mid) / mid
    if abs(dev) <= 0.10:
        verdict, msg = "FAIR", "Price is within fair market range."
    elif abs(dev) <= 0.25:
        d = "above" if dev > 0 else "below"
        verdict, msg = "REVIEW", f"Price is {abs(dev)*100:.0f}% {d} market value. Consider reviewing."
    else:
        d = "above" if dev > 0 else "below"
        verdict, msg = "FLAGGED", f"Price is {abs(dev)*100:.0f}% {d} market value. Flagged for review."
    return {"verdict":verdict,"message":msg,"listed_price_usd":listed_price_usd,
            "predicted_price_usd":mid,"price_range_low_usd":r["price_range_low_usd"],
            "price_range_high_usd":r["price_range_high_usd"],"deviation_pct":round(dev*100,1)}

class BatchRequest(BaseModel):
    properties: list[HouseFeatures]

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if len(request.properties) > 50:
        raise HTTPException(status_code=400, detail="Max 50 properties per batch.")
    return {"results":[predict(h) for h in request.properties],"count":len(request.properties)}
