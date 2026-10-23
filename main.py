import pandas as pd

# Load the sales dataset
file_path = "dataset/Sales_Data_2022_2025.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# -----------------------------
# STEP 2: DATA CLEANING
# -----------------------------

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Check data types
print("\nData types after cleaning:")
print(df.dtypes)

# Final dataset information
print("\nCleaned dataset shape:")
print(df.shape)

# Save cleaned dataset
output_file = "outputs/Cleaned_Sales_Data.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", output_file)
# -----------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# -----------------------------

print("\n===== SALES SUMMARY =====")

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()

print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Quantity Sold:", total_quantity)

print("\nSales Statistics:")
print(df["Sales"].describe())
# -----------------------------
# STEP 3.2: SALES BY YEAR AND MONTH
# -----------------------------

print("\n===== SALES BY YEAR =====")

yearly_sales = df.groupby("Year")["Sales"].sum()

print(yearly_sales)

print("\n===== SALES BY MONTH =====")

monthly_sales = df.groupby("Month")["Sales"].sum()

print(monthly_sales)
# -----------------------------
# STEP 3.3: SALES BY PRODUCT AND CATEGORY
# -----------------------------

print("\n===== SALES BY PRODUCT =====")

product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)

print(product_sales)

print("\n===== SALES BY CATEGORY =====")

category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

print(category_sales)
# -----------------------------
# STEP 3.4: SALES BY REGION AND SALES CHANNEL
# -----------------------------

print("\n===== SALES BY REGION =====")

region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

print(region_sales)

print("\n===== SALES BY SALES CHANNEL =====")

channel_sales = df.groupby("Sales_Channel")["Sales"].sum().sort_values(ascending=False)

print(channel_sales)
# -----------------------------
# STEP 4: DATA VISUALIZATION
# STEP 4.1: YEAR-WISE SALES
# -----------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

yearly_sales.plot(kind="bar")

plt.title("Year-wise Sales")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/Year_Wise_Sales.png")
plt.show()

print("\nYear-wise sales graph saved successfully!")
# -----------------------------
# STEP 4.2: MONTH-WISE SALES
# -----------------------------

plt.figure(figsize=(10, 5))

monthly_sales.plot(kind="line", marker="o")

plt.title("Month-wise Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/Month_Wise_Sales.png")
plt.show()

print("\nMonth-wise sales graph saved successfully!")
# -----------------------------
# STEP 4.3: PRODUCT-WISE SALES
# -----------------------------

plt.figure(figsize=(10, 6))

product_sales.plot(kind="bar")

plt.title("Product-wise Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/Product_Wise_Sales.png")
plt.show()

print("\nProduct-wise sales graph saved successfully!")
# -----------------------------
# STEP 4.4: CATEGORY-WISE SALES
# -----------------------------

plt.figure(figsize=(8, 5))

category_sales.plot(kind="bar")

plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/Category_Wise_Sales.png")
plt.show()

print("\nCategory-wise sales graph saved successfully!")
# -----------------------------
# STEP 4.5: REGION-WISE SALES
# -----------------------------

plt.figure(figsize=(9, 5))

region_sales.plot(kind="bar")

plt.title("Region-wise Sales")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/Region_Wise_Sales.png")
plt.show()

print("\nRegion-wise sales graph saved successfully!")
# -----------------------------
# STEP 4.7: CATEGORY-WISE SALES PIE CHART
# -----------------------------

plt.figure(figsize=(8, 8))

category_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sales Distribution by Category")
plt.ylabel("")
plt.tight_layout()

plt.savefig("outputs/Category_Sales_Pie.png")
plt.show()

print("\nCategory-wise sales pie chart saved successfully!")
# -----------------------------
# STEP 5: PREPARE MONTHLY SALES DATA
# -----------------------------

print("\n===== MONTHLY SALES DATA FOR FORECASTING =====")

# Create monthly sales
monthly_forecast = df.groupby(
    df["Date"].dt.to_period("M")
)["Sales"].sum().reset_index()

# Convert period to date
monthly_forecast["Date"] = monthly_forecast["Date"].dt.to_timestamp()

print(monthly_forecast)

print("\nNumber of months:", len(monthly_forecast))

# Save monthly forecasting dataset
monthly_forecast.to_csv(
    "outputs/Monthly_Sales_Forecasting_Data.csv",
    index=False
)

print("\nMonthly forecasting dataset saved successfully!")

# -----------------------------
# STEP 6: LINEAR REGRESSION MODEL
# -----------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n===== LINEAR REGRESSION MODEL =====")

# Create time index
monthly_forecast["Time_Index"] = range(len(monthly_forecast))

# Features and target
X = monthly_forecast[["Time_Index"]]
y = monthly_forecast["Sales"]

# Chronological train-test split
train_size = int(len(monthly_forecast) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Create model
linear_model = LinearRegression()

# Train model
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

print("\nLinear Regression model trained successfully!")

# -----------------------------
# STEP 7: RANDOM FOREST MODEL
# -----------------------------

from sklearn.ensemble import RandomForestRegressor

print("\n===== RANDOM FOREST MODEL =====")

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R² Score:", round(rf_r2, 4))

print("\nRandom Forest model trained successfully!")
# -----------------------------
# STEP 8: CREATE TIME FEATURES
# -----------------------------

print("\n===== CREATING TIME FEATURES =====")

# Create time-based features
monthly_forecast["Year"] = monthly_forecast["Date"].dt.year
monthly_forecast["Month"] = monthly_forecast["Date"].dt.month
monthly_forecast["Quarter"] = monthly_forecast["Date"].dt.quarter

print("\nMonthly forecasting data with new features:")
print(monthly_forecast.head())

print("\nTime features created successfully!")
# -----------------------------
# STEP 9: IMPROVED LINEAR REGRESSION
# -----------------------------

print("\n===== IMPROVED LINEAR REGRESSION =====")

# Features and target
X_improved = monthly_forecast[
    ["Time_Index", "Year", "Month", "Quarter"]
]

y_improved = monthly_forecast["Sales"]

# Chronological train-test split
train_size = int(len(monthly_forecast) * 0.8)

X_train_improved = X_improved.iloc[:train_size]
X_test_improved = X_improved.iloc[train_size:]

y_train_improved = y_improved.iloc[:train_size]
y_test_improved = y_improved.iloc[train_size:]

# Create model
improved_linear_model = LinearRegression()

# Train model
improved_linear_model.fit(
    X_train_improved,
    y_train_improved
)
import joblib

# Save trained model
joblib.dump(
    improved_linear_model,
    "models/sales_forecasting_model.pkl"
)

print("\nTrained model saved successfully!")
print("File: models/sales_forecasting_model.pkl")

improved_pred = improved_linear_model.predict(X_test_improved)

# Predictions
improved_pred = improved_linear_model.predict(
    X_test_improved
)

# Evaluation
improved_mae = mean_absolute_error(
    y_test_improved,
    improved_pred
)

improved_rmse = np.sqrt(
    mean_squared_error(
        y_test_improved,
        improved_pred
    )
)

improved_r2 = r2_score(
    y_test_improved,
    improved_pred
)

print("\nImproved Linear Regression Results:")
print("MAE:", round(improved_mae, 2))
print("RMSE:", round(improved_rmse, 2))
print("R² Score:", round(improved_r2, 4))

print("\nImproved Linear Regression trained successfully!")
# -----------------------------
# STEP 10: IMPROVED RANDOM FOREST
# -----------------------------

print("\n===== IMPROVED RANDOM FOREST =====")

# Features and target
X_rf_improved = monthly_forecast[
    ["Time_Index", "Year", "Month", "Quarter"]
]

y_rf_improved = monthly_forecast["Sales"]

# Chronological train-test split
X_train_rf = X_rf_improved.iloc[:train_size]
X_test_rf = X_rf_improved.iloc[train_size:]

y_train_rf = y_rf_improved.iloc[:train_size]
y_test_rf = y_rf_improved.iloc[train_size:]

# Create model
improved_rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
improved_rf_model.fit(
    X_train_rf,
    y_train_rf
)

# Predictions
improved_rf_pred = improved_rf_model.predict(
    X_test_rf
)

# Evaluation
improved_rf_mae = mean_absolute_error(
    y_test_rf,
    improved_rf_pred
)

improved_rf_rmse = np.sqrt(
    mean_squared_error(
        y_test_rf,
        improved_rf_pred
    )
)

improved_rf_r2 = r2_score(
    y_test_rf,
    improved_rf_pred
)

print("\nImproved Random Forest Results:")
print("MAE:", round(improved_rf_mae, 2))
print("RMSE:", round(improved_rf_rmse, 2))
print("R² Score:", round(improved_rf_r2, 4))

print("\nImproved Random Forest trained successfully!")

# -----------------------------
# STEP 11: CREATE LAG FEATURES
# -----------------------------

print("\n===== CREATING LAG FEATURES =====")

# Previous month's sales
monthly_forecast["Sales_Lag_1"] = monthly_forecast["Sales"].shift(1)

# Sales from the same month of the previous year
monthly_forecast["Sales_Lag_12"] = monthly_forecast["Sales"].shift(12)

print("\nLag features created:")
print(
    monthly_forecast[
        ["Date", "Sales", "Sales_Lag_1", "Sales_Lag_12"]
    ].head(15)
)

print("\nLag features created successfully!")
# -----------------------------
# STEP 12: PREPARE LAG DATA
# -----------------------------

print("\n===== PREPARING LAG DATA =====")

# Remove rows where lag values are missing
lag_data = monthly_forecast.dropna(
    subset=["Sales_Lag_1", "Sales_Lag_12"]
).copy()

print("\nLag dataset:")
print(
    lag_data[
        ["Date", "Sales", "Sales_Lag_1", "Sales_Lag_12"]
    ].head()
)

print("\nLag dataset shape:", lag_data.shape)

print("\nLag data prepared successfully!")

# -----------------------------
# STEP 13: LAG-BASED LINEAR REGRESSION
# -----------------------------

print("\n===== LAG-BASED LINEAR REGRESSION =====")

# Create time index for lag dataset
lag_data["Time_Index"] = range(len(lag_data))

# Features
X_lag = lag_data[
    [
        "Time_Index",
        "Year",
        "Month",
        "Quarter",
        "Sales_Lag_1",
        "Sales_Lag_12"
    ]
]

# Target
y_lag = lag_data["Sales"]

# Chronological train-test split
lag_train_size = int(len(lag_data) * 0.8)

X_lag_train = X_lag.iloc[:lag_train_size]
X_lag_test = X_lag.iloc[lag_train_size:]

y_lag_train = y_lag.iloc[:lag_train_size]
y_lag_test = y_lag.iloc[lag_train_size:]

# Create model
lag_linear_model = LinearRegression()

# Train
lag_linear_model.fit(
    X_lag_train,
    y_lag_train
)

# Predict
lag_linear_pred = lag_linear_model.predict(
    X_lag_test
)

# Evaluation
lag_linear_mae = mean_absolute_error(
    y_lag_test,
    lag_linear_pred
)

lag_linear_rmse = np.sqrt(
    mean_squared_error(
        y_lag_test,
        lag_linear_pred
    )
)

lag_linear_r2 = r2_score(
    y_lag_test,
    lag_linear_pred
)

print("\nLag-Based Linear Regression Results:")
print("MAE:", round(lag_linear_mae, 2))
print("RMSE:", round(lag_linear_rmse, 2))
print("R² Score:", round(lag_linear_r2, 4))

print("\nLag-Based Linear Regression trained successfully!")
# -----------------------------
# STEP 14: MODEL COMPARISON
# -----------------------------

print("\n===== MODEL COMPARISON =====")

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Improved Linear Regression",
        "Improved Random Forest",
        "Lag-Based Linear Regression"
    ],
    "MAE": [
        mae,
        rf_mae,
        improved_mae,
        improved_rf_mae,
        lag_linear_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse,
        improved_rmse,
        improved_rf_rmse,
        lag_linear_rmse
    ],
    "R2_Score": [
        r2,
        rf_r2,
        improved_r2,
        improved_rf_r2,
        lag_linear_r2
    ]
})

print(model_comparison)

# Save comparison table
model_comparison.to_csv(
    "outputs/Model_Comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
# -----------------------------
# STEP 15: NEXT 12 MONTH FORECAST
# -----------------------------

print("\n===== NEXT 12 MONTH SALES FORECAST =====")

# Create future dates
last_date = monthly_forecast["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

# Create future dataframe
future_data = pd.DataFrame({
    "Date": future_dates
})

# Create time features
future_data["Time_Index"] = range(
    len(monthly_forecast),
    len(monthly_forecast) + 12
)

future_data["Year"] = future_data["Date"].dt.year
future_data["Month"] = future_data["Date"].dt.month
future_data["Quarter"] = future_data["Date"].dt.quarter

# Create prediction features
X_future = future_data[
    ["Time_Index", "Year", "Month", "Quarter"]
]

# Generate forecast
future_data["Forecasted_Sales"] = (
    improved_linear_model.predict(X_future)
)

# Display forecast
print("\n12-Month Sales Forecast:")
print(
    future_data[
        ["Date", "Forecasted_Sales"]
    ]
)

# Save forecast
future_data.to_csv(
    "outputs/Next_12_Month_Sales_Forecast.csv",
    index=False
)

print("\n12-month forecast saved successfully!")
# -----------------------------
# STEP 16: HISTORICAL + FORECAST GRAPH
# -----------------------------

print("\n===== HISTORICAL + FORECAST GRAPH =====")

plt.figure(figsize=(12, 6))

# Historical sales
plt.plot(
    monthly_forecast["Date"],
    monthly_forecast["Sales"],
    marker="o",
    label="Historical Sales"
)

# Forecasted sales
plt.plot(
    future_data["Date"],
    future_data["Forecasted_Sales"],
    marker="o",
    linestyle="--",
    label="Forecasted Sales"
)

plt.title("Historical Sales and 12-Month Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/Historical_and_Forecast_Sales.png"
)

plt.show()

print("\nHistorical and forecast graph saved successfully!")
# -----------------------------
# STEP 17: 2026 FORECAST GRAPH
# -----------------------------

print("\n===== 2026 FORECAST GRAPH =====")

plt.figure(figsize=(10, 5))

plt.plot(
    future_data["Date"],
    future_data["Forecasted_Sales"],
    marker="o",
    linewidth=2
)

plt.title("2026 Monthly Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Forecasted Sales")

plt.grid(True)
plt.xticks(
    future_data["Date"],
    future_data["Date"].dt.strftime("%b 2026"),
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "outputs/2026_Monthly_Sales_Forecast.png"
)

plt.show()

print("\n2026 forecast graph saved successfully!")
# -----------------------------
# STEP 18: FORECAST SUMMARY
# -----------------------------

print("\n===== 2026 FORECAST SUMMARY =====")

total_forecast = future_data["Forecasted_Sales"].sum()
average_forecast = future_data["Forecasted_Sales"].mean()

highest_index = future_data["Forecasted_Sales"].idxmax()
lowest_index = future_data["Forecasted_Sales"].idxmin()

highest_month = future_data.loc[highest_index, "Date"]
highest_sales = future_data.loc[highest_index, "Forecasted_Sales"]

lowest_month = future_data.loc[lowest_index, "Date"]
lowest_sales = future_data.loc[lowest_index, "Forecasted_Sales"]

jan_sales = future_data.iloc[0]["Forecasted_Sales"]
dec_sales = future_data.iloc[-1]["Forecasted_Sales"]

growth_percentage = (
    (dec_sales - jan_sales) / jan_sales
) * 100

print("Total Forecasted Sales:", round(total_forecast, 2))
print("Average Monthly Forecast:", round(average_forecast, 2))

print(
    "Highest Forecast Month:",
    highest_month.strftime("%B %Y")
)
print("Highest Forecasted Sales:", round(highest_sales, 2))

print(
    "Lowest Forecast Month:",
    lowest_month.strftime("%B %Y")
)
print("Lowest Forecasted Sales:", round(lowest_sales, 2))

print(
    "January to December Growth:",
    round(growth_percentage, 2),
    "%"
)

print("\nForecast summary calculated successfully!")

# -----------------------------
# STEP 19: SAVE FORECAST SUMMARY
# -----------------------------

print("\n===== SAVING FORECAST SUMMARY =====")

forecast_summary = pd.DataFrame({
    "Metric": [
        "Total Forecasted Sales",
        "Average Monthly Forecast",
        "Highest Forecast Month",
        "Highest Forecasted Sales",
        "Lowest Forecast Month",
        "Lowest Forecasted Sales",
        "January to December Growth"
    ],
    "Value": [
        round(total_forecast, 2),
        round(average_forecast, 2),
        highest_month.strftime("%B %Y"),
        round(highest_sales, 2),
        lowest_month.strftime("%B %Y"),
        round(lowest_sales, 2),
        round(growth_percentage, 2)
    ]
})

forecast_summary.to_csv(
    "outputs/Forecast_Summary_2026.csv",
    index=False
)

print("\nForecast summary saved successfully!")
print("File: outputs/Forecast_Summary_2026.csv")
# -----------------------------
# STEP 20: COMBINE ACTUAL AND FORECAST DATA
# -----------------------------

print("\n===== COMBINING ACTUAL AND FORECAST DATA =====")

# Historical data
actual_data = monthly_forecast[
    ["Date", "Sales"]
].copy()

actual_data["Type"] = "Actual"
actual_data.rename(
    columns={"Sales": "Sales_Value"},
    inplace=True
)

# Forecast data
forecast_data = future_data[
    ["Date", "Forecasted_Sales"]
].copy()

forecast_data["Type"] = "Forecast"
forecast_data.rename(
    columns={"Forecasted_Sales": "Sales_Value"},
    inplace=True
)

# Combine both datasets
combined_sales = pd.concat(
    [actual_data, forecast_data],
    ignore_index=True
)

print("\nCombined actual and forecast data:")
print(combined_sales.tail(15))

print("\nCombined dataset shape:")
print(combined_sales.shape)

# Save combined dataset
combined_sales.to_csv(
    "outputs/Actual_and_Forecast_Sales.csv",
    index=False
)

print("\nActual and forecast data saved successfully!")
print("File: outputs/Actual_and_Forecast_Sales.csv")
# -----------------------------
# STEP 21: ACTUAL VS FORECAST GRAPH
# -----------------------------

print("\n===== ACTUAL VS FORECAST GRAPH =====")

actual_plot = combined_sales[
    combined_sales["Type"] == "Actual"
]

forecast_plot = combined_sales[
    combined_sales["Type"] == "Forecast"
]

plt.figure(figsize=(12, 6))

# Actual sales
plt.plot(
    actual_plot["Date"],
    actual_plot["Sales_Value"],
    marker="o",
    label="Actual Sales"
)

# Forecast sales
plt.plot(
    forecast_plot["Date"],
    forecast_plot["Sales_Value"],
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

plt.title("Actual Sales vs 2026 Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/Actual_vs_Forecast_Sales.png"
)

plt.show()

print("\nActual vs forecast graph saved successfully!")
# -----------------------------
# STEP 22: FINAL PROJECT RESULTS
# -----------------------------

print("\n===== FINAL PROJECT RESULTS =====")

final_results = pd.DataFrame({
    "Metric": [
        "Total Historical Sales (2022-2025)",
        "Total Historical Profit (2022-2025)",
        "Total Quantity Sold",
        "Historical Records",
        "Historical Months",
        "Forecast Months",
        "2026 Total Forecasted Sales",
        "2026 Average Monthly Forecast",
        "2026 Lowest Forecast Month",
        "2026 Highest Forecast Month",
        "January to December Growth",
        "Best Model by Test MAE"
    ],
    "Value": [
        round(total_sales, 2),
        round(total_profit, 2),
        total_quantity,
        len(df),
        len(monthly_forecast),
        len(future_data),
        round(total_forecast, 2),
        round(average_forecast, 2),
        lowest_month.strftime("%B %Y"),
        highest_month.strftime("%B %Y"),
        round(growth_percentage, 2),
        "Improved Linear Regression"
    ]
})

print("\n===== FINAL RESULTS TABLE =====")
print(final_results)

final_results.to_csv(
    "outputs/Final_Project_Results.csv",
    index=False
)

print("\nFinal project results saved successfully!")
print("File: outputs/Final_Project_Results.csv")