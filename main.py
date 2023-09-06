from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random

app = FastAPI()

data = {
    "Asset": [f"Asset_{i}" for i in range(1, 1001)],  # Asset as a string
    "Country": [random.choice(["South Africa", "Ghana", "Mozambique", "Kenya", "China", "America", "Australia"]) for _ in range(1000)],  # Country as a string
    "Price": [round(random.uniform(1, 1000000), 2) for _ in range(1000)],  # Price as a float
    "Term": [random.randint(1, 10) for _ in range(1000)],  # Term as an integer
    "Political_Stability_Index": [random.randint(1, 1000) for _ in range(1000)],  # Political_Stability_Index as a number between 1 and 100
    "Category": [random.choice(["Treasury Bill", "Coupon", "Bonds", "Loans"]) for _ in range(1000)],
}

# Sample training asset data (replace with your training dataset)
training_asset_data = pd.DataFrame(data)

data2 = {
    "Asset": [f"Asset_{i}" for i in range(1, 101)],  # Asset as a string
    "Country": [random.choice(["South Africa", "Ghana", "Mozambique", "Kenya", "China", "America", "Australia"]) for _ in range(100)],  # Country as a string
    "Price": [round(random.uniform(1, 1000000), 2) for _ in range(100)],  # Price as a float
    "Term": [random.randint(1, 10) for _ in range(100)],  # Term as an integer
    "Political_Stability_Index": [random.randint(1, 100) for _ in range(100)],  # Political_Stability_Index as a number between 1 and 100
    "Category": [random.choice(["Treasury Bill", "Coupon", "Bonds", "Loans"]) for _ in range(100)],
}

# Sample current asset data (replace with your current asset dataset)
current_asset_data = pd.DataFrame(data2)

# Function to train the linear regression model for asset risk assessment
def train_linear_regression(data):
    X = data[['Term']].values
    y = data['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to calculate risk score with political stability factor
def calculate_risk_score(predicted_price, transaction_value, political_stability):
    return abs(predicted_price - transaction_value) * (1 - political_stability / 100)


# Function to optimize collateral and get the cheapest asset with value
def optimize_collateral(model, transaction_value, transaction_term, asset_data, country, political_stability):
    asset_data['Predicted_Price'] = model.predict(np.array([[transaction_term]]))[0]
    
    # Calculate risk scores for each asset with political stability factor
    asset_data['Risk_Score'] = asset_data.apply(
        # lambda row: calculate_risk_score(row['Predicted_Price'], transaction_value, row['Political_Stability_Index']),
        lambda row: calculate_risk_score(row['Predicted_Price'], transaction_value, random.randint(1, 100)),
        axis=1
    )
    
    asset_data = asset_data.sort_values(by=['Risk_Score'])
    
    if len(asset_data) > 0:
        cheapest_asset = asset_data.iloc[0]['Asset']
        transaction_value = asset_data.iloc[0]['Predicted_Price']
        risk_score = asset_data.iloc[0]['Risk_Score']
    else:
        cheapest_asset = 'No eligible asset available'
        transaction_value = 0
    
    return cheapest_asset, transaction_value, risk_score

def summarize_assets_by_category(asset_data):
    summary = asset_data.groupby('Category').agg(
        Total_Price=pd.NamedAgg(column='Price', aggfunc='sum'),
        Average_Price=pd.NamedAgg(column='Price', aggfunc='mean'),
        Min_Price=pd.NamedAgg(column='Price', aggfunc='min'),
        Max_Price=pd.NamedAgg(column='Price', aggfunc='max')
    ).reset_index()
    
    return summary
# Pydantic model for request data
class TransactionRequest(BaseModel):
    transaction_value: float
    transaction_term: int
    country: str
    new_country_features: list  # Replace with your features for the new country

@app.post("/predict_asset")
async def predict_asset(transaction_data: TransactionRequest):
    # Train the linear regression model for asset risk assessment using the training dataset
    trained_model = train_linear_regression(training_asset_data)
    
    # Predict the political index for the new country (replace with your actual model)
    # predicted_political_index = predict_political_index(transaction_data.new_country_features)
    predicted_political_index = random.randint(1, 100)
    
    # Optimize collateral and get the cheapest asset with value
    cheapest_asset, transaction_value, risk_score = optimize_collateral(
        trained_model, transaction_data.transaction_value, transaction_data.transaction_term,
        current_asset_data, transaction_data.country, predicted_political_index
    )
    
    category_summary = summarize_assets_by_category(current_asset_data)
    
    
    return {"recommended_asset": cheapest_asset, "transaction_value": transaction_value, "category_summary": category_summary.to_dict(orient='records'), "Score": risk_score}
