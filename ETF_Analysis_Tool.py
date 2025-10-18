import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.ticker as mticker

plt.style.use("seaborn-v0_8-darkgrid")

def fetch_latest_prices(tickers):
    """Fetch latest closing price for each ticker."""
    data = yf.download(tickers, period="1d", interval="1d", progress=False)
    latest = {}
    for t in tickers:
        try:
            latest_price = data['Close'][t].iloc[-1]
        except Exception:
            latest_price = np.nan
        latest[t] = latest_price
    return latest

def fetch_historical(tickers, start_date, end_date):
    """Fetch historical daily closing prices."""
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return df

def compute_etf_value(latest_prices, weights):
    """Compute ETF value = sum(weight * price)."""
    value = sum(latest_prices[t] * weights[t] for t in weights)
    return value

def build_etf_history(historical_df, weights):
    """Combine weighted stock prices to produce ETF value series."""
    etf = pd.DataFrame(index=historical_df.index)
    etf['ETF_Price'] = sum(historical_df[t] * weights[t] for t in weights)
    return etf

def forecast_linear(etf_series, forecast_days=756):
    """Linear regression forecast for given # of trading days (~3 years)."""
    etf = etf_series.dropna()
    X = np.arange(len(etf)).reshape(-1,1)
    y = etf.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(etf), len(etf)+forecast_days).reshape(-1,1)
    y_pred = model.predict(future_X)

    last_date = etf.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=forecast_days+1)[1:]
    df_forecast = pd.DataFrame({'Forecast': y_pred.flatten()}, index=future_dates)

    r2 = r2_score(y, model.predict(X))
    return model, df_forecast, r2

def plot_etf(etf_hist, forecast_df, model, r2):
    """Visualize ETF history, regression fit, and long-term forecast."""
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(etf_hist.index, etf_hist['ETF_Price'], label='Historical', linewidth=2.5)
    ax.plot(forecast_df.index, forecast_df['Forecast'], linestyle='--', label='Forecast (3 yrs)', linewidth=2.2)

    # Trend line overlay for historical segment
    X = np.arange(len(etf_hist)).reshape(-1,1)
    trend = model.predict(X)
    ax.plot(etf_hist.index, trend, label='Trend Line', color='orange', alpha=0.8)

    # Formatting and labels
    ax.set_title("ETF Value & Linear Forecast", fontsize=15, pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("ETF Value", fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    # Model info (upper center)
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    ax.text(0.5, 0.95,
            f"y = {coef:.4f}x + {intercept:.2f}   |   R² = {r2:.3f}",
            ha='center', va='top',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

def main():
    # === Customize your ETF ===
    tickers = ['AAPL', 'TSLA', 'META', 'AMZN']
    weights = {'AAPL':0.35, 'TSLA':0.25, 'META':0.25, 'AMZN':0.15} # Weights sosum to 1.0

    # === Yahoo Finance latest & historical data ===
    print(f"Fetching live data for: {', '.join(tickers)} ...")
    latest_prices = fetch_latest_prices(tickers)
    etf_value = compute_etf_value(latest_prices, weights)
    print(f"\nCurrent ETF Value: ${etf_value:,.2f}")
    for t, p in latest_prices.items():
        print(f"  {t}: ${p:.2f} (weight {weights[t]*100:.1f}%)")

    start_date = '2015-01-01'
    end_date = dt.date.today().isoformat()
    hist = fetch_historical(tickers, start_date, end_date)
    etf_hist = build_etf_history(hist, weights)

    # === Forecast using applied regression ===
    print("\nPerforming linear regression forecast (3 years)...")
    model, forecast_df, r2 = forecast_linear(etf_hist['ETF_Price'], forecast_days=756)

    # === Final summary info-sheet ===
    start_val, end_val = etf_hist['ETF_Price'].iloc[0], etf_hist['ETF_Price'].iloc[-1]
    growth_pct = (end_val / start_val - 1) * 100
    print(f"\nHistorical Range: {etf_hist.index[0].date()} → {etf_hist.index[-1].date()}")
    print(f"Value change: ${start_val:,.2f} → ${end_val:,.2f} ({growth_pct:.1f}%)")
    print(f"Projected ETF value in 3 years: ${forecast_df['Forecast'].iloc[-1]:,.2f}")

    # === Graph the visual ===
    plot_etf(etf_hist, forecast_df, model, r2)

if __name__ == "__main__":
    main()