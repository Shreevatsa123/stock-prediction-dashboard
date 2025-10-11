import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from groq import Groq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import finnhub

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Financial Analyst Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def get_stock_data(ticker, period="20y"):
    """Downloads historical stock data from Yahoo Finance."""
    return yf.Ticker(ticker).history(period=period)

@st.cache_data
def get_news_sentiment(company_name):
    """Scrapes Google News and performs sentiment analysis."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = f"https://www.google.com/search?q={company_name}+stock+news&tbm=nws"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [item.get_text() for item in soup.find_all('div', {'class': 'n0jPhd ynAwRc MBeuO nDgy9d'}, limit=5)]
    
    if not headlines:
        return "Could not retrieve news headlines."
    
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = sentiment_analyzer(headlines)
    
    analysis = f"News Sentiment Analysis for {company_name}:\n"
    for i, headline in enumerate(headlines):
        analysis += f"- Headline: {headline}\n  - Sentiment: {results[i]['label']} (Score: {results[i]['score']:.2f})\n"
    return analysis

# Make sure to add this import at the top of your file

# Make sure to add this import at the top of your file
import xgboost as xgb

@st.cache_data
def train_prediction_model(ticker):
    """Creates advanced features and trains an XGBoost model on a 7-year period."""
    # --- UPGRADE 1: Focus on a more relevant 7-year period ---
    data = get_stock_data(ticker, period="7y")

    # --- Get and Process News Sentiment (as before) ---
    news_df = get_historical_news(ticker, data)
    if not news_df.empty:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment = sentiment_analyzer(news_df['headline'].tolist())
        news_df['sentiment_score'] = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiment]
        daily_sentiment = news_df.resample('D')['sentiment_score'].mean()
        data = data.join(daily_sentiment)
        data['sentiment_score'].fillna(0, inplace=True)
    else:
        data['sentiment_score'] = 0

    # --- UPGRADE 2: Add more advanced features ---
    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Std_Dev_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['Std_Dev_20'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Std_Dev_20'] * 2)
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(14).mean()

    data['RSI'] = 100 - (100 / (1 + ((data['Close'].diff().where(data['Close'].diff() > 0, 0)).rolling(window=14).mean() / 
                                      (-data['Close'].diff().where(data['Close'].diff() < 0, 0)).rolling(window=14).mean())))
    
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    # Define the new, expanded feature set
    features = [
        'Close', 'Volume', 'sentiment_score', 'RSI', 
        'SMA_20', 'Upper_Band', 'Lower_Band', 'ATR'
    ]
    X = data[features]
    y = data['Target']
    
    # Time-series split remains the same
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    # --- UPGRADE 3: Use the more powerful XGBoost model ---
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, accuracy, data, features

@st.cache_data
def get_historical_news(ticker, data):
    """Fetches historical news from Finnhub for the date range of the stock data."""
    st.write("Fetching historical news... This may take a minute for the first run.")
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
    all_news = []
    
    # Get date range from the main stock dataframe
    start_date = data.index.min().strftime('%Y-%m-%d')
    end_date = data.index.max().strftime('%Y-%m-%d')
    
    # Fetch news
    news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    all_news.extend(news)
    
    # Convert to DataFrame
    news_df = pd.DataFrame(all_news)
    if not news_df.empty:
        news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
        news_df.set_index('datetime', inplace=True)
        news_df = news_df.tz_localize('UTC').tz_convert('America/New_York')
    return news_df

def generate_ai_summary(ticker, company_name, model_accuracy, news_sentiment):
    """Uses an LLM to generate a final summary report."""
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    system_prompt = """
    You are a senior financial analyst AI. Your task is to write a concise, easy-to-understand investment briefing.
    You will be given a predictive model's accuracy and a news sentiment analysis.
    Synthesize this information into a final report with two sections:
    1. **Predictive Model Outlook:** State the model's accuracy and explain what it means in simple terms.
    2. **News & Sentiment Outlook:** Summarize the sentiment of recent news and what it implies.
    Conclude with a one-sentence summary. Do not include any investment advice.
    """
    human_prompt = f"""
    Please generate a report for {company_name} ({ticker}) based on the following data:
    --- PREDICTIVE MODEL ACCURACY ---
    {model_accuracy:.2%}
    --- NEWS SENTIMENT DATA ---
    {news_sentiment}
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ],
        model="llama-3.3-70b-versatile", 
    )
    return chat_completion.choices[0].message.content

# --- Streamlit App UI ---
st.title("AI Financial Analyst Dashboard")

# Sidebar for User Input
st.sidebar.header("User Input")
ticker_map = {
    "NVIDIA": "NVDA", "Apple": "AAPL", "Google": "GOOGL", 
    "Microsoft": "MSFT", "Amazon": "AMZN", "Tesla": "TSLA"
}
company_name = st.sidebar.selectbox("Select a Company", list(ticker_map.keys()))
ticker = ticker_map[company_name]

if ticker:
    # Train the predictive model and get results
    with st.spinner(f"Training predictive model for {company_name}..."):
        model, accuracy, data, features = train_prediction_model(ticker)
    
    st.header(f"Predictive Analysis for {company_name}")
    st.metric(label="Prediction Model Accuracy", value=f"{accuracy:.2%}")
    st.info("This model attempts to predict if the stock price will go up or down the next day based on historical data.")
    
    # Display visualizations
    st.subheader("Historical Closing Price")
    st.line_chart(data['Close'])
    
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)
    st.pyplot(fig)

    # AI Agent Summary Section
    st.header("AI-Generated Summary Report")
    if st.button("Generate AI Summary"):
        with st.spinner("Getting news sentiment and generating report..."):
            news_sentiment = get_news_sentiment(company_name)
            ai_summary = generate_ai_summary(ticker, company_name, accuracy, news_sentiment)
            st.markdown(ai_summary)