import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# --- 1. CONFIGURATION ---
DATA_PATH = "Data/Walmart.csv"
OUTPUT_DASHBOARD = "Data/sales_forecast_dashboard.csv"
OUTPUT_IMPORTANCE = "Data/feature_importance.csv"

def run_pipeline():
    print("🚀 Starting Walmart Sales Analysis Pipeline...")

    # --- 2. DATA LOADING & CLEANING ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Store', 'Date'])
    print(f"✅ Data loaded: {len(df)} records.")

    # --- 3. FEATURE ENGINEERING ---
    print("🛠️ Engineering features...")
    # Lags and Rolling Averages
    df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Sales_Rolling_4'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(window=4).mean())
    
    # Temporal features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Seasonal flags
    df['Is_Summer'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    df['Is_Holiday_Season'] = df['Month'].apply(lambda x: 1 if x in [11, 12] else 0)

    # Drop rows with NaNs from lags/rolling
    df_model = df.dropna().copy()

    # --- 4. SQL KPI ANALYSIS ---
    print("🗄️ Running SQL KPI Analysis...")
    conn = sqlite3.connect(':memory:')
    df.to_sql('sales', conn, index=False, if_exists='replace')

    # Total Revenue
    total_rev = pd.read_sql_query("SELECT SUM(Weekly_Sales) as total FROM sales", conn).iloc[0]['total']
    # Holiday Lift
    holiday_lift = pd.read_sql_query("""
        SELECT Holiday_Flag, AVG(Weekly_Sales) as avg_sales 
        FROM sales GROUP BY Holiday_Flag
    """, conn)
    
    print(f"💰 Total Revenue: ${total_rev:,.2f}")
    print("📊 Holiday Performance:")
    print(holiday_lift)
    conn.close()

    # --- 5. MACHINE LEARNING ---
    print("🤖 Training Random Forest Model...")
    features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                'Sales_Lag_1', 'Sales_Rolling_4', 'Month', 'Year', 'WeekOfYear', 'Is_Summer', 'Is_Holiday_Season']
    X = df_model[features]
    y = df_model['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"📈 Model Performance: MAE=${mae:,.2f}, R²={r2:.4f}")

    # --- 6. PREPARE OUTPUTS ---
    print("💾 Saving outputs for Dashboard...")
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(OUTPUT_IMPORTANCE, index=False)

    # Dashboard CSV
    dashboard_df = df_model.copy()
    dashboard_df['Predicted_Sales'] = np.nan
    dashboard_df.iloc[-len(preds):, dashboard_df.columns.get_loc('Predicted_Sales')] = preds
    dashboard_df['Set'] = 'Train'
    dashboard_df.iloc[-len(preds):, dashboard_df.columns.get_loc('Set')] = 'Test'
    
    dashboard_df = dashboard_df.rename(columns={'Weekly_Sales': 'Actual_Sales'})
    dashboard_df['Residual'] = dashboard_df['Actual_Sales'] - dashboard_df['Predicted_Sales']
    
    final_cols = ['Date', 'Store', 'Actual_Sales', 'Predicted_Sales', 'Set', 'Residual', 'Holiday_Flag']
    dashboard_df[final_cols].to_csv(OUTPUT_DASHBOARD, index=False)

    print("✅ Pipeline completed successfully.")
    print(f"📍 Dashboard data: {OUTPUT_DASHBOARD}")
    print(f"📍 Feature importance: {OUTPUT_IMPORTANCE}")

if __name__ == "__main__":
    run_pipeline()
