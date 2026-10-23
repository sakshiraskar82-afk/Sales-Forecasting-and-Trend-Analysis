import joblib
import pandas as pd

# Load trained model
model = joblib.load(
    "models/sales_forecasting_model.pkl"
)

print("Sales forecasting model loaded successfully!")

# Create future dates
future_dates = pd.date_range(
    start="2026-01-01",
    periods=12,
    freq="MS"
)

# Create prediction data
future_data = pd.DataFrame({
    "Date": future_dates
})

# Create the same features used during training
future_data["Time_Index"] = range(48, 60)
future_data["Year"] = future_data["Date"].dt.year
future_data["Month"] = future_data["Date"].dt.month
future_data["Quarter"] = future_data["Date"].dt.quarter

# Select model features
X_future = future_data[
    ["Time_Index", "Year", "Month", "Quarter"]
]

# Make predictions
future_data["Predicted_Sales"] = model.predict(X_future)

# Display predictions
print("\n===== 2026 SALES PREDICTION =====")

print(
    future_data[
        ["Date", "Predicted_Sales"]
    ].to_string(index=False)
)

# Save predictions
future_data.to_csv(
    "outputs/Predictions_From_Saved_Model.csv",
    index=False
)

print("\nPredictions saved successfully!")
print(
    "File: outputs/Predictions_From_Saved_Model.csv"
)