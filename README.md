- 👋 Hi, I’m @saeedsaeedi1990
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
saeedsaeedi1990/saeedsaeedi1990 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import ta

app = Flask(__name__)

def fetch_data(symbol):
    data = yf.download(symbol, period="30d", interval="1h")
    data.dropna(inplace=True)

    # محاسبه RSI و MACD
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["Signal"] = macd.macd_signal()

    return data.tail(50)

@app.route("/")
def index():
    btc = fetch_data("BTC-USD")
    eth = fetch_data("ETH-USD")

    return render_template("index.html", btc=btc.tail(10).to_dict(orient="records"), 
                                         eth=eth.tail(10).to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)unzip *.zipcripto-project.zipgit add .
git commit -m "توضیح کوتاه درباره تغییرات"
git push origin mainmainstreamlit run app.pypip install streamlitrm file-8xFcyxGqUFBEp9ZVTWA1M1.zipunzip file-8xFcyxGqUFBEp9ZVTWA1M1.zipimport streamlit as st

st.title("تحلیل ارز دیجیتال")
st.write("سلام! این پروژه تحلیل بیت‌کوین و اتریومه. به زودی ویژگی‌های حرفه‌ای اضافه می‌شه.")import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

st.title("Crypto Dashboard - Bitcoin & Ethereum")

# انتخاب ارز
crypto_options = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
selected_crypto = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# گرفتن داده‌ها
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, period="6mo", interval="1d")
    data.dropna(inplace=True)
    return data

df = get_data(symbol)

# نمایش قیمت
st.subheader(f"{selected_crypto} Price Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'],
                             name='Candlestick'))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# محاسبه RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

st.subheader("RSI Indicator")
st.line_chart(rsi)

# محاسبه MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()

st.subheader("MACD Indicator")
macd_df = pd.DataFrame({"MACD": macd, "Signal": signal})
st.line_chart(macd_df)import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

st.title("Crypto Dashboard - Bitcoin & Ethereum")

# انتخاب ارز
crypto_options = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
selected_crypto = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# گرفتن داده‌ها
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, period="6mo", interval="1d")
    data.dropna(inplace=True)
    return data

df = get_data(symbol)

# نمایش قیمت
st.subheader(f"{selected_crypto} Price Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'],
                             name='Candlestick'))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# محاسبه RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

st.subheader("RSI Indicator")
st.line_chart(rsi)

# محاسimport streamlit as st
from data import get_crypto_data
from analysis import calculate_indicators
from sentiment import get_sentiment
from history import get_price_history
from database import save_to_db, load_from_db

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Crypto Dashboard")

coin = st.sidebar.selectbox("Select a cryptocurrency", ["Bitcoin", "Ethereum"])
days = st.sidebar.slider("Days of data", 7, 90, 30)

df = get_crypto_data(coin, days)
if df is not None:
    df = calculate_indicators(df)

    st.subheader(f"{coin} Price Chart")
    st.line_chart(df['Close'])

    st.subheader("Technical Indicators")
    st.line_chart(df[['RSI', 'MACD']])

    st.subheader("Market Sentiment")
    sentiment = get_sentiment(coin)
    st.write(f"Sentiment score for {coin}: {sentiment}")

    if st.button("Save data to database"):
        save_to_db(df, coin)
        st.success("Data saved successfully!")

    if st.button("Load saved data"):
        saved_df = load_from_db(coin)
        st.write(saved_df.tail())

else:
    st.error("Failed to fetch data. Please check your API key or connection.")git remote add origin https://github.com/saeedsaeedi1990/crypto_dashboard.git
git branch -M main
git push -u origin maincrypto_dashboard/
├── main.py
├── requirements.txt
├── run_dashboard.sh
├── README.md
├── utils/
│   ├── data.py
│   ├── analysis.py
│   ├── sentiment.py
│   └── database.py
└── pages/
    ├── dashboard.py
    ├── settings.py
    ├── sentiment_page.py
    └── history.pyDescription (optional)import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("تحلیل ارز دیجیتال - بیت‌کوین و اتریوم")

coin = st.selectbox("انتخاب ارز:", ["Bitcoin (BTC)", "Ethereum (ETH)"])
symbol = "BTC-USD" if "BTC" in coin else "ETH-USD"

# Load data
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period='90d', interval='1d')
    data.dropna(inplace=True)
    return data

df = load_data(symbol)

# Indicators
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data):
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal

df['RSI'] = compute_rsi(df)
df['MACD'], df['Signal'] = compute_macd(df)

# Plotting
st.subheader("نمودار قیمت و اندیکاتورها")

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax[0].plot(df['Close'], label='قیمت', color='blue')
ax[0].set_title('قیمت')

ax[1].plot(df['RSI'], label='RSI', color='green')
ax[1].axhline(70, color='red', linestyle='--')
ax[1].axhline(30, color='red', linestyle='--')
ax[1].set_title('RSI')

ax[2].plot(df['MACD'], label='MACD', color='purple')
ax[2].plot(df['Signal'], label='Signal', color='orange')
ax[2].set_title('MACD')
ax[2].legend()

st.pyplot(fig)

# Export
st.subheader("دانلود دیتا")
csv = df.to_csv().encode('utf-8')
st.download_button("دانلود CSV", data=csv, file_name=f'{symbol}_analysis.csv', mime='text/csv')#!/data/data/com.termux/files/usr/bin/bash

# به‌روزرسانی و نصب ابزارهای لازم
pkg update -y
pkg upgrade -y
pkg install -y python wget unzip git

# نصب pip و streamlit
pip install --upgrade pip
pip install streamlit

# مسیر پروژه
PROJECT_PATH="/sdcard/Download/crypto_dashboard_advanced"

# رفتن به مسیر پروژه
cd "$PROJECT_PATH" || {
  echo "مسیر پروژه پیدا نشد!"
  exit 1
}

# نصب کتابخانه‌ها از requirements.txt
pip install -r requirements.txt

# اجرای streamlit در پس‌زمینه
echo "در حال اجرای Streamlit..."
nohup streamlit run main.py &

# اگر ngrok نصب نشده، نصب کن
if [ ! -f "$PROJECT_PATH/ngrok" ]; then
  echo "در حال دانلود Ngrok..."
  wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-arm.zip
  unzip ngrok-stable-linux-arm.zip
  chmod +x ngrok
fi

# اگر auth token تنظیم نشده بود، راهنمایی کن
if ! grep -q "authtoken" ~/.ngrok2/ngrok.yml 2>/dev/null; then
  echo "توکن Ngrok تنظیم نشده!"
  echo "لطفا این دستور رو یک بار دستی بزن:"
  echo "./ngrok config add-authtoken <توکن تو از ngrok.com>"
  exit 1
fi

# اجرای ngrok
echo "در حال اجرای Ngrok..."
./ngrok http 8501pkg update
pkg install python
pip install --upgrade pippkg update
pkg upgrade
pkg install python
pkg install git
pip install --upgrade piphttp://localhost:8501cd path/to/crypto_dashboard_advancedpip install -r requirements.txthttps://huggingface.co/spaces/your-username/crypto-dashboard# file: dashboard.py

import streamlit as st
from technical_analysis import get_historical_data, add_indicators, plot_chart

# پیکربندی صفحه
st.set_page_config(page_title="Crypto Analyzer", layout="wide")

st.title("داشبورد تحلیل ارز دیجیتال")

# انتخاب ارز
coin = st.selectbox("ارز مورد نظر:", ["bitcoin", "ethereum"])
# انتخاب بازه زمانی
days = st.slider("بازه زمانی (روز):", min_value=7, max_value=90, value=30)

# گرفتن و پردازش داده
with st.spinner("در حال دریافت داده‌ها..."):
    df = get_historical_data(coin, "usd", days)
    df = add_indicators(df)

# نمایش جدول و نمودار
st.subheader("نمودار قیمت و میانگین‌های متحرک")
plot_chart(df)  # تابع Plotly داخل فایل قبلی

st.dataframe(df.tail(10))crypto_dashboard/
│
├── dashboard.py             ← فایل اصلی Streamlit
├── historical_data.py       ← دریافت داده از CoinGecko
├── technical_analysis.py    ← محاسبه اندیکاتورها
├── sentiment.py             ← تحلیل احساسات ساده
└── requirements.txt         ← لیست کتابخانه‌ها# file: technical_analysis.py

import pandas as pd
import plotly.graph_objs as go
from historical_data import get_historical_data  # از فایل قبلی ایمپورت می‌کنیم

def add_indicators(df):
    df["SMA_7"] = df["price"].rolling(window=7).mean()
    df["EMA_7"] = df["price"].ewm(span=7, adjust=False).mean()
    return df

def plot_chart(df):
    fig = go.Figure()

    # قیمت
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines", name="Price"))
    # SMA
    fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_7"], mode="lines", name="SMA 7-day"))
    # EMA
    fig.add_trace(go.Scatter(x=df["date"], y=df["EMA_7"], mode="lines", name="EMA 7-day"))

    fig.update_layout(
        title="Bitcoin Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    fig.show()

if __name__ == "__main__":
    df = get_historical_data("bitcoin", "usd", 30)
    df = add_indicators(df)
    plot_chart(df)pip install requests pandas streamlit plotly# file: historical_data.py

import requests
import pandas as pd
from datetime import datetime

def get_historical_data(coin_id="bitcoin", currency="usd", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": currency,
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()

    prices = data["prices"]  # [timestamp, price]
    
    # تبدیل به DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df = df[["date", "price"]]
    return df

if __name__ == "__main__":
    df = get_historical_data("bitcoin", "usd", 30)
    print(df)# file: crypto_data.py

import requests
import pandas as pd

def get_price(coin_id="bitcoin", currency="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": currency,
        "include_24hr_change": "true"
    }
    response = requests.get(url, params=params)
    data = response.json()
    price = data[coin_id][currency]
    change = data[coin_id][f"{currency}_24h_change"]
    return price, change

if __name__ == "__main__":
    coins = ["bitcoin", "ethereum"]
    for coin in coins:
        price, change = get_price(coin)
        print(f"{coin.capitalize()} - قیمت: {price} دلار | تغییر ۲۴ ساعته: {change:.2f}%")
