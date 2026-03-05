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
print(f"  ABC Real Estate - Model R2: {metadata['r2_test']*100:.1f}%")
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
    print(f"\n  {label}")
    print(f"  Predicted : ${r['predicted']:,}  |  Range: ${r['low']:,} - ${r['high']:,}")
    print(f"  Listed    : ${listed:,}  ->  {r.get('verdict','')}")

print("\n  All tests done!")
print("\nNext step - start the API:")
print("  uvicorn api.main:app --reload --port 8000")
