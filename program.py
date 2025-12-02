# -------------------------------------------------------------
#  RANDOM FOREST REGRESSION – SLV PRICE PREDICTION
#  Author      : Anish S Nair
#  Adm. No     : 25MM2546
#  Reg. No     : KTE25CSDC02
#  Dept        : Mechanical Engineering
#  College     : RIT Kottayam
# -------------------------------------------------------------

import math
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

DATASET = "data.csv"

MODEL_OUTPUT = "rf_slv_model.joblib"
COMPARISON_CSV = "actual_vs_predicted.csv"
IMPORTANCES_CSV = "feature_importances.csv"

HEATMAP_PNG = "correlation_heatmap.png"
ACTVSPRED_PNG = "actual_vs_predicted_plot.png"
SCATTER_PNG = "actual_vs_predicted_scatter.png"
RESIDUALS_PNG = "residuals_hist.png"
TOPFEATURES_PNG = "top_features.png"

os.makedirs("plots", exist_ok=True)

df = pd.read_csv(DATASET)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print("\nDataset preview (first 5 rows):")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe().T)

corr = df.drop(columns=['Date']).corr()
plt.figure(figsize=(6,5))
plt.title("Correlation Matrix")
plt.imshow(corr, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join("plots", HEATMAP_PNG))
plt.close()

data = df.copy()
data.set_index('Date', inplace=True)

data['year'] = data.index.year
data['month'] = data.index.month
data['day'] = data.index.day
data['weekday'] = data.index.weekday
data['dayofyear'] = data.index.dayofyear

for lag in [1,2,3,5,10]:
    data[f'slv_lag_{lag}'] = data['SLV'].shift(lag)

data['slv_roll_3'] = data['SLV'].rolling(window=3, min_periods=1).mean()
data['slv_roll_7'] = data['SLV'].rolling(window=7, min_periods=1).mean()

data.ffill(inplace=True)
data.bfill(inplace=True)

target_col = 'SLV'
feature_cols = [c for c in data.columns if c != target_col]

X = data[feature_cols].copy()
y = data[target_col].copy()

split_idx = int(len(X)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTotal rows: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=1)
grid.fit(X_train_scaled, y_train)
best = grid.best_estimator_
print("\nBest params:", grid.best_params_)

y_pred = best.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print(f"\nEvaluation -- R2: {r2:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, EVS: {evs:.6f}")

comparison = pd.DataFrame({
    'Date': X_test.index,
    'SLV_actual': y_test.values,
    'SLV_predicted': y_pred
}).set_index('Date')
comparison.to_csv(COMPARISON_CSV)

plt.figure(figsize=(10,4))
plt.plot(comparison.index, comparison['SLV_actual'], label='Actual')
plt.plot(comparison.index, comparison['SLV_predicted'], label='Predicted')
plt.xlabel("Date")
plt.ylabel("SLV Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("plots", ACTVSPRED_PNG))
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SLV")
plt.ylabel("Predicted SLV")
plt.tight_layout()
plt.savefig(os.path.join("plots", SCATTER_PNG))
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30)
plt.xlabel("Residuals")
plt.tight_layout()
plt.savefig(os.path.join("plots", RESIDUALS_PNG))
plt.close()

importances = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.reset_index().rename(columns={'index':'feature', 0:'importance'}).to_csv(IMPORTANCES_CSV, index=False)

plt.figure(figsize=(8,5))
top_features = importances.head(10)
plt.barh(top_features.index, top_features.values)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join("plots", TOPFEATURES_PNG))
plt.close()

joblib.dump(best, MODEL_OUTPUT)

summary = {
    "rows_total": len(X),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "best_params": grid.best_params_,
    "r2": r2,
    "rmse": rmse,
    "mae": mae,
    "evs": evs
}
print("\nSummary:", summary)

print("\n--- SLV PRICE PREDICTION FOR NEW USER INPUT ---")

try:
    spx = float(input("Enter SPX value: "))
    gld = float(input("Enter GLD value: "))
    uso = float(input("Enter USO value: "))
    eurusd = float(input("Enter EUR/USD value: "))

    last_row = data.iloc[-1]

    new_data = pd.DataFrame({
        'SPX': [spx],
        'GLD': [gld],
        'USO': [uso],
        'EUR/USD': [eurusd],
        'year': [last_row['year']],
        'month': [last_row['month']],
        'day': [last_row['day']],
        'weekday': [last_row['weekday']],
        'dayofyear': [last_row['dayofyear']],
        'slv_lag_1': [last_row['SLV']],
        'slv_lag_2': [last_row['slv_lag_1']],
        'slv_lag_3': [last_row['slv_lag_2']],
        'slv_lag_5': [last_row['slv_lag_3']],
        'slv_lag_10': [last_row['slv_lag_5']],
        'slv_roll_3': [last_row['slv_roll_3']],
        'slv_roll_7': [last_row['slv_roll_7']],
    })

    new_data_scaled = new_data.copy()
    new_data_scaled[num_cols] = scaler.transform(new_data[num_cols])

    pred_value = best.predict(new_data_scaled)[0]
    print(f"\nPredicted SLV Price: {pred_value:.4f}")

except Exception as e:
    print("Error:", e)
