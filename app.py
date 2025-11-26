from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Build supervised ML dataset
def build_supervised(close_prices, window=10):
    X, y = [], []
    for i in range(len(close_prices) - window):
        X.append(close_prices[i:i + window])
        y.append(close_prices[i + window])
    return np.array(X), np.array(y)


# News sentiment (VADER)
def get_news_sentiment(symbol):
    API_KEY = "YOUR_API_KEY"  # <-- Insert your NewsAPI key
    analyzer = SentimentIntensityAnalyzer()

    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}"
        articles = requests.get(url).json().get("articles", [])
    except:
        return "Unavailable", []

    headlines = []
    score_sum = 0

    for a in articles[:5]:
        title = a.get("title", "")
        score = analyzer.polarity_scores(title)["compound"]
        score_sum += score
        headlines.append({"title": title, "sentiment": round(score, 3)})

    if not headlines:
        return "Unavailable", []

    avg = score_sum / len(headlines)

    label = "Positive" if avg > 0.2 else "Negative" if avg < -0.2 else "Neutral"
    return label, headlines


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    forecast_list = None
    error = None

    # extras
    signal_label = None
    signal_desc = None
    signal_change_pct = None
    sentiment_label = None
    sentiment_text = None
    news_sentiment_label = None
    news_headlines = None
    metric_mae = None
    metric_mape = None

    # chart data
    historical_labels = None
    historical_values = None
    forecast_labels = None
    forecast_values = None
    candlestick_data = None

    if request.method == "POST":

        stock = request.form.get("stock_name", "").upper().strip()
        days_raw = request.form.get("days", "5")

        # auto append Indian suffix
        if not stock.endswith(".NS") and not stock.endswith(".BO"):
            stock = stock + ".NS"

        try:
            days = int(days_raw)
        except:
            error = "Invalid days input."
            return render_template("index.html", error=error)

        try:
            df = yf.download(stock, period="2y", interval="1d")

            if df.empty:
                raise Exception("Invalid stock or no data found.")

            df = df.dropna()
            dates = df.index
            close = df["Close"].astype(float).values.flatten()

            # ---------------- CANDLESTICK FIXED ----------------
            candlestick_data = []
            for ts, row in df.iterrows():
                candlestick_data.append({
                    "x": int(ts.timestamp() * 1000),
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"])
                })

            # Need at least 40 days for ML
            if len(close) < 40:
                raise Exception("Not enough historical data.")

            # Build ML dataset
            window = 10
            X, y = build_supervised(close, window)
            split = int(len(X) * 0.8)

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = RandomForestRegressor(n_estimators=220, random_state=42)
            model.fit(X_train, y_train)

            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                err = np.abs(y_test - y_pred)
                metric_mae = float(np.mean(err))
                metric_mape = float(np.mean(err / np.maximum(y_test, 1e-6)) * 100)

            # Forecast
            history = list(close[-window:])
            preds = []
            for _ in range(days):
                x_in = np.array(history[-window:]).reshape(1, -1)
                next_val = model.predict(x_in)[0]
                preds.append(float(next_val))
                history.append(next_val)

            prediction = round(preds[-1], 2)
            forecast_values = [round(p, 2) for p in preds]

            # Future dates
            last_date = dates[-1].date()
            forecast_labels = []
            forecast_list = []

            for i, p in enumerate(forecast_values):
                fut = last_date + timedelta(days=i + 1)
                ds = fut.strftime("%Y-%m-%d")
                forecast_labels.append(ds)
                forecast_list.append({"date": ds, "price": p})

            # Historical labels
            historical_labels = [d.date().strftime("%Y-%m-%d") for d in dates]
            historical_values = [round(float(v), 2) for v in close]

            # Signals
            last_real = close[-1]
            next_pred = preds[0]
            pct = (next_pred - last_real) / last_real * 100
            signal_change_pct = round(pct, 2)

            if pct > 2:
                signal_label = "BUY"
                signal_desc = "Upside movement expected."
            elif pct < -2:
                signal_label = "SELL"
                signal_desc = "Possible downside movement."
            else:
                signal_label = "HOLD"
                signal_desc = "Sideways market expected."

            # Technical sentiment
            recent = close[-11:]
            returns = (recent[1:] - recent[:-1]) / recent[:-1]
            avg_ret = np.mean(returns)

            if avg_ret > 0.004:
                sentiment_label = "Bullish"
                sentiment_text = "Momentum is strong."
            elif avg_ret < -0.004:
                sentiment_label = "Bearish"
                sentiment_text = "Momentum is weakening."
            else:
                sentiment_label = "Neutral"
                sentiment_text = "Market is stable."

            # News sentiment
            news_sentiment_label, news_headlines = get_news_sentiment(stock)

        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,

        # Signals
        signal_label=signal_label,
        signal_desc=signal_desc,
        signal_change_pct=signal_change_pct,

        # Sentiments
        sentiment_label=sentiment_label,
        sentiment_text=sentiment_text,
        news_sentiment_label=news_sentiment_label,
        news_headlines=news_headlines,

        # Accuracy
        metric_mae=metric_mae,
        metric_mape=metric_mape,

        # Chart data
        historical_labels=historical_labels,
        historical_values=historical_values,
        forecast_labels=forecast_labels,
        forecast_values=forecast_values,
        candlestick_data=candlestick_data
    )


if __name__ == "__main__":
    app.run(debug=True)
