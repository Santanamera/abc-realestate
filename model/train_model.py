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
print(f"\nLoaded {len(df)} records")
print(f"Price range: ${df['price_usd'].min():,} - ${df['price_usd'].max():,}")

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

print("\nTraining model...")
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nResults:")
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

print("\nModel saved -> model/house_price_model.pkl")
print("Done!")
