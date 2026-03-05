"""
ABC Real Estate — One-click project setup.
Run this from inside your abc_realestate folder:
    python setup.py
"""

import os

# ── Folder structure ──────────────────────────────────────────────────────────
for folder in ["data", "model", "api"]:
    os.makedirs(folder, exist_ok=True)

print("📁 Folders created")

# ── File contents ─────────────────────────────────────────────────────────────

files = {}

# ── requirements.txt ──────────────────────────────────────────────────────────
files["requirements.txt"] = """\
fastapi==0.110.0
uvicorn[standard]==0.29.0
scikit-learn==1.4.2
pandas==2.2.1
numpy==1.26.4
joblib==1.3.2
pydantic==2.6.4
"""

# ── data/generate_data.py ─────────────────────────────────────────────────────
files["data/generate_data.py"] = '''\
"""
ABC Real Estate - Rwanda Housing Dataset Generator
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

districts = {
    "Gasabo":     {"base": 180_000, "weight": 0.30},
    "Kicukiro":   {"base": 160_000, "weight": 0.25},
    "Nyarugenge": {"base": 200_000, "weight": 0.20},
    "Musanze":    {"base":  90_000, "weight": 0.10},
    "Rubavu":     {"base":  80_000, "weight": 0.08},
    "Huye":       {"base":  70_000, "weight": 0.07},
}

district_names   = list(districts.keys())
district_bases   = [districts[d]["base"]   for d in district_names]
district_weights = [districts[d]["weight"] for d in district_names]

chosen_districts = np.random.choice(district_names, size=N, p=district_weights)
base_prices = np.array([districts[d]["base"] for d in chosen_districts], dtype=float)

bedrooms   = np.random.choice([1,2,3,4,5], size=N, p=[0.05,0.25,0.40,0.20,0.10])
bathrooms  = np.clip(bedrooms - np.random.choice([0,1], size=N, p=[0.6,0.4]), 1, 5)
size_sqm   = (bedrooms * 30 + np.random.normal(0, 15, N)).clip(40, 400).astype(int)
year_built = np.random.randint(1990, 2024, N)
age        = 2024 - year_built

property_types = np.random.choice(
    ["Apartment","Standalone Villa","Townhouse","Bungalow"],
    size=N, p=[0.35,0.25,0.25,0.15]
)
type_multiplier = {
    "Apartment":0.85,"Standalone Villa":1.30,"Townhouse":1.00,"Bungalow":0.90
}

has_garage          = np.random.choice([0,1], size=N, p=[0.55,0.45])
has_garden          = np.random.choice([0,1], size=N, p=[0.50,0.50])
has_swimming_pool   = np.random.choice([0,1], size=N, p=[0.85,0.15])
has_security        = np.random.choice([0,1], size=N, p=[0.40,0.60])
is_gated_community  = np.random.choice([0,1], size=N, p=[0.65,0.35])
distance_to_cbd_km  = np.round(np.random.exponential(scale=8, size=N).clip(0.5,40),1)
road_condition      = np.random.choice(["Tarmac","Gravel","Dirt"], size=N, p=[0.55,0.30,0.15])
road_mult           = {"Tarmac":1.10,"Gravel":1.00,"Dirt":0.88}

price = (
    base_prices
    * np.array([type_multiplier[t] for t in property_types])
    * np.array([road_mult[r] for r in road_condition])
    + bedrooms  * 8_000
    + bathrooms * 4_000
    + size_sqm  * 300
    - age       * 400
    + has_garage        * 12_000
    + has_garden        * 6_000
    + has_swimming_pool * 25_000
    + has_security      * 5_000
    + is_gated_community * 15_000
    - distance_to_cbd_km * 1_500
    + np.random.normal(0, 10_000, N)
).clip(30_000, 900_000)

price = np.round(price, -2).astype(int)

df = pd.DataFrame({
    "district":           chosen_districts,
    "property_type":      property_types,
    "bedrooms":           bedrooms,
    "bathrooms":          bathrooms,
    "size_sqm":           size_sqm,
    "year_built":         year_built,
    "has_garage":         has_garage,
    "has_garden":         has_garden,
    "has_swimming_pool":  has_swimming_pool,
    "has_security":       has_security,
    "is_gated_community": is_gated_community,
    "distance_to_cbd_km": distance_to_cbd_km,
    "road_condition":     road_condition,
    "price_usd":          price,
})

df.to_csv("data/rwanda_housing.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df[["district","property_type","bedrooms","size_sqm","price_usd"]].head())
'''

# ── model/train_model.py ──────────────────────────────────────────────────────
files["model/train_model.py"] = '''\
"""
ABC Real Estate - Model Training Pipeline
"""
import pandas as pd
import numpy as np
import joblib, json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  ABC Real Estate - Model Training")
print("=" * 55)

df = pd.read_csv("data/rwanda_housing.csv")
print(f"\\nLoaded {len(df)} records")
print(f"Price range: ${df[\'price_usd\'].min():,} - ${df[\'price_usd\'].max():,}")

TARGET      = "price_usd"
CATEGORICAL = ["district","property_type","road_condition"]
NUMERICAL   = [
    "bedrooms","bathrooms","size_sqm","year_built",
    "has_garage","has_garden","has_swimming_pool",
    "has_security","is_gated_community","distance_to_cbd_km"
]

X = df[CATEGORICAL + NUMERICAL]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Split: {len(X_train)} train / {len(X_test)} test")

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ("num", StandardScaler(), NUMERICAL),
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=5, random_state=42, n_jobs=-1
    ))
])

print("\\nTraining model...")
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\\nResults:")
print(f"  R2   : {r2:.4f}  ({r2*100:.1f}% accuracy)")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")

cv = cross_val_score(model_pipeline, X, y, cv=5, scoring="r2")
print(f"  CV R2: {cv.mean():.4f} +/- {cv.std():.4f}")

joblib.dump(model_pipeline, "model/house_price_model.pkl")
json.dump({
    "model_type":"RandomForestRegressor",
    "features_categorical": CATEGORICAL,
    "features_numeric": NUMERICAL,
    "target": TARGET,
    "r2_test": round(r2,4),
    "mae_test": round(mae,2),
    "rmse_test": round(rmse,2),
}, open("model/model_metadata.json","w"), indent=2)

print("\\nModel saved -> model/house_price_model.pkl")
print("Done!")
'''

# ── api/__init__.py ───────────────────────────────────────────────────────────
files["api/__init__.py"] = ""

# ── api/main.py ───────────────────────────────────────────────────────────────
files["api/main.py"] = '''\
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
'''

# ── test_api_logic.py ─────────────────────────────────────────────────────────
files["test_api_logic.py"] = '''\
"""
ABC Real Estate - Quick test (no server needed)
Run: python test_api_logic.py
"""
import joblib, json
import pandas as pd

model    = joblib.load("model/house_price_model.pkl")
metadata = json.load(open("model/model_metadata.json"))
MARGIN   = 0.10

def predict(house, listed_price=None):
    df   = pd.DataFrame([house])
    pred = float(model.predict(df)[0])
    low, high, mid = round(pred*(1-MARGIN),-2), round(pred*(1+MARGIN),-2), round(pred,-2)
    result = {"predicted":int(mid),"low":int(low),"high":int(high)}
    if listed_price:
        dev = (listed_price - mid) / mid
        direction = "ABOVE" if dev > 0 else "BELOW"
        if abs(dev) > 0.25:
            result["verdict"] = f"FLAGGED - {abs(dev)*100:.0f}% {direction} fair value"
        elif abs(dev) > 0.10:
            result["verdict"] = f"REVIEW  - {abs(dev)*100:.0f}% {direction} fair value"
        else:
            result["verdict"] = "FAIR"
    return result

print("=" * 55)
print(f"  ABC Real Estate - Model R2: {metadata[\'r2_test\']*100:.1f}%")
print("=" * 55)

tests = [
    ("Luxury Villa - Nyarugenge", 380_000, {
        "district":"Nyarugenge","property_type":"Standalone Villa","road_condition":"Tarmac",
        "bedrooms":4,"bathrooms":3,"size_sqm":220,"year_built":2020,
        "has_garage":1,"has_garden":1,"has_swimming_pool":1,
        "has_security":1,"is_gated_community":1,"distance_to_cbd_km":2.5}),
    ("Small Apartment - Musanze", 45_000, {
        "district":"Musanze","property_type":"Apartment","road_condition":"Gravel",
        "bedrooms":1,"bathrooms":1,"size_sqm":45,"year_built":2010,
        "has_garage":0,"has_garden":0,"has_swimming_pool":0,
        "has_security":0,"is_gated_community":0,"distance_to_cbd_km":15.0}),
    ("Overpriced Townhouse - should be FLAGGED", 600_000, {
        "district":"Kicukiro","property_type":"Townhouse","road_condition":"Tarmac",
        "bedrooms":3,"bathrooms":2,"size_sqm":130,"year_built":2016,
        "has_garage":1,"has_garden":0,"has_swimming_pool":0,
        "has_security":1,"is_gated_community":0,"distance_to_cbd_km":7.0}),
]

for label, listed, house in tests:
    r = predict(house, listed)
    print(f"\\n  {label}")
    print(f"  Predicted : ${r[\'predicted\']:,}  |  Range: ${r[\'low\']:,} - ${r[\'high\']:,}")
    print(f"  Listed    : ${listed:,}  ->  {r.get(\'verdict\',\'\')}")

print("\\n  All tests done!")
print("\\nNext step - start the API:")
print("  uvicorn api.main:app --reload --port 8000")
'''

# ── Write all files ───────────────────────────────────────────────────────────
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n" + "=" * 55)
print("  Setup complete! All files created.")
print("=" * 55)
print("""
Next steps - run these commands one by one:

  1.  python -m pip install -r requirements.txt
  2.  python data/generate_data.py
  3.  python model/train_model.py
  4.  python test_api_logic.py
  5.  uvicorn api.main:app --reload --port 8000

Then open:  http://localhost:8000/docs
""")
