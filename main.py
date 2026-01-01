from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="California Housing Price Predictor")

# Load the trained model once at startup
print("Loading model...")
with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✓ Model loaded successfully!")

# Cache HTML content in memory
html_cache = None

def get_html():
    global html_cache
    if html_cache is None:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_cache = f.read()
    return html_cache

# Define input schema
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housingMedianAge: float
    totalRooms: float
    totalBedrooms: float
    population: float
    households: float
    medianIncome: float
    oceanProximity: str

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    print("⚠️ Warning: 'static' directory not found")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return get_html()

@app.post("/predict")
async def predict(input_data: HousingInput):
    """
    Predict house price based on input features
    Returns prediction in dollars
    """
    try:
        # Create a dataframe with the input data
        input_dict = {
            'longitude': [input_data.longitude],
            'latitude': [input_data.latitude],
            'housing_median_age': [input_data.housingMedianAge],
            'total_rooms': [input_data.totalRooms],
            'total_bedrooms': [input_data.totalBedrooms],
            'population': [input_data.population],
            'households': [input_data.households],
            'median_income': [input_data.medianIncome],
            'ocean_proximity': [input_data.oceanProximity]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # IMPORTANT: Create the engineered features that were used in training
        input_df['rooms_per_household'] = input_df['total_rooms'] / input_df['households']
        input_df['bedrooms_per_rooms'] = input_df['total_bedrooms'] / input_df['total_rooms']
        input_df['population_per_household'] = input_df['population'] / input_df['households']
        
        # Make prediction (model returns log scale)
        prediction_log = model.predict(input_df)[0]
        
        # Convert back to dollars (inverse of log1p transformation)
        prediction_dollars = np.expm1(prediction_log)
        
        return {
            "success": True,
            "predicted_price": round(prediction_dollars, 2),
            "price_formatted": f"${prediction_dollars:,.2f}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "✓ Model is running successfully!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)