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
