import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load or download S&P 500 data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0, parse_dates=True)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

# Ensure index is datetime
sp500.index = pd.to_datetime(sp500.index, utc=True)

# Display the data
print(sp500)

# Remove unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create target variable
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Filter data from 1990 onwards
sp500 = sp500.loc["1990-01-01":].copy()

# Initial model training
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

# Make predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
print(f"Initial precision score: {precision_score(test['Target'], preds)}")


# Define prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# Define backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


# Run initial backtest
predictions = backtest(sp500, model, predictors)
print(f"\nPredictions value counts:\n{predictions['Predictions'].value_counts()}")
print(f"Precision score: {precision_score(predictions['Target'], predictions['Predictions'])}")
print(f"\nTarget distribution:\n{predictions['Target'].value_counts() / predictions.shape[0]}")


# Create new features
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

# Drop rows with NaN values
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
print(f"\nDataframe after feature engineering:\n{sp500}")

# Improved model with new features
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Run backtest with new predictors
predictions = backtest(sp500, model, new_predictors)
print(f"\nFinal predictions value counts:\n{predictions['Predictions'].value_counts()}")
print(f"Final precision score: {precision_score(predictions['Target'], predictions['Predictions'])}")
print(f"\nFinal target distribution:\n{predictions['Target'].value_counts() / predictions.shape[0]}")
print(f"\nFinal predictions:\n{predictions}")
