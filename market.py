import yfinance as yf
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler


# Load or download S&P 500 data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0, parse_dates=True)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

# Ensure index is datetime
sp500.index = pd.to_datetime(sp500.index, utc=True)

# Remove unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create target variable
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Filter data from 1990 onwards
sp500 = sp500.loc["1990-01-01":].copy()


# -------------------------
# PREDICT & BACKTEST
# -------------------------

def predict(train, test, predictors, model, scaler=None):
    X_train = train[predictors]
    X_test  = test[predictors]

    # Apply scaling (only for SVM and KNN)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, train["Target"])

    # RandomForest and KNN have predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # For SVM with probability=True
        probs = model.decision_function(X_test)
        # Normalize decision function to [0,1]
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)

    preds = (probs >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")

    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250, scaler=None):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model, scaler)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


# -------------------------
# FEATURE ENGINEERING
# -------------------------

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]


# RSI 14
window = 14
delta = sp500["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window).mean()
avg_loss = loss.rolling(window).mean()

rs = avg_gain / avg_loss
sp500["RSI14"] = 100 - (100 / (1 + rs))

new_predictors.append("RSI14")


# Drop rows with NaN values
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])


# -------------------------
# MODELS TO TEST
# -------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, min_samples_split=50, random_state=1
    ),
    "SVM": SVC(kernel="rbf", probability=True),  # probability=True enables predict_proba
    "KNN": KNeighborsClassifier(n_neighbors=10)
}

# SVM & KNN need scaling
scalers = {
    "RandomForest": None,
    "SVM": StandardScaler(),
    "KNN": StandardScaler()
}

# -------------------------
# RUN BACKTESTS
# -------------------------
for name, model in models.items():
    print(f"\n===== Running {name} =====")

    scaler = scalers[name]
    predictions = backtest(sp500, model, new_predictors, scaler=scaler)

    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"{name} precision score: {precision}")
    print(predictions["Predictions"].value_counts())
