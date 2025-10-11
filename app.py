# app.py - Main Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import finnhub
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from groq import Groq
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst Dashboard", page_icon="🤖", layout="wide")

# --- Initialize Session State ---
# This dictionary will hold all our logs and data for the details page
if 'run_details' not in st.session_state:
    st.session_state.run_details = {}

# --- Caching Functions (remain the same) ---
@st.cache_data
def get_stock_data(ticker, period="7y"):
    return yf.Ticker(ticker).history(period=period)

@st.cache_data
def get_historical_news(ticker, data):
    # (Your existing get_historical_news function code goes here - no changes needed)
    st.write("Fetching historical news... This may take a minute for the first run.")
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
    all_news = []
    start_date = data.index.min().strftime('%Y-%m-%d')
    end_date = data.index.max().strftime('%Y-%m-%d')
    news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    all_news.extend(news)
    news_df = pd.DataFrame(all_news)
    if not news_df.empty:
        news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
        news_df.set_index('datetime', inplace=True)
        news_df = news_df.tz_localize('UTC').tz_convert('America/New_York')
    return news_df

# --- MODIFIED Training Function ---
def train_prediction_model(ticker):
    """Trains model and logs all intermediate steps to session_state."""
    logs = []
    st.session_state.run_details = {} # Clear previous run details

    # 1. Data Collection
    logs.append("Step 1: Data Collection")
    data = get_stock_data(ticker, period="7y")
    logs.append(f"-> Fetched {len(data)} data points of historical stock data.")
    st.session_state.run_details['initial_data_head'] = data.head()

    # 2. News and Sentiment
    logs.append("\nStep 2: News & Sentiment Analysis")
    news_df = get_historical_news(ticker, data)
    if not news_df.empty:
        # (Your sentiment analysis logic is here)
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment = sentiment_analyzer(news_df['headline'].tolist())
        news_df['sentiment_score'] = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiment]
        daily_sentiment = news_df.resample('D')['sentiment_score'].mean()
        data = data.join(daily_sentiment)
        data['sentiment_score'].fillna(0, inplace=True)
        logs.append(f"-> Analyzed {len(news_df)} news headlines and merged sentiment scores.")
    else:
        data['sentiment_score'] = 0
        logs.append("-> No historical news found. Using neutral sentiment score.")

    # 3. Feature Engineering
    logs.append("\nStep 3: Feature Engineering")
    # (Your feature engineering logic is here)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Std_Dev_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['Std_Dev_20'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Std_Dev_20'] * 2)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + ((data['Close'].diff().where(data['Close'].diff() > 0, 0)).rolling(window=14).mean() / (-data['Close'].diff().where(data['Close'].diff() < 0, 0)).rolling(window=14).mean())))
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    logs.append("-> Created features: SMA, Bollinger Bands, ATR, RSI.")
    st.session_state.run_details['featured_data_head'] = data.head()

    # 4. Data Splitting & EDA
    logs.append("\nStep 4: Data Splitting & EDA")
    features = ['Close', 'Volume', 'sentiment_score', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'ATR']
    X = data[features]
    y = data['Target']
    
    correlation_matrix = X.corr()
    st.session_state.run_details['correlation_matrix'] = correlation_matrix

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    logs.append(f"-> Performed TimeSeriesSplit. Training data shape: {X_train.shape}. Testing data shape: {X_test.shape}.")

    # 5. Hyperparameter Tuning
    logs.append("\nStep 5: Hyperparameter Tuning with GridSearchCV")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=tscv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logs.append(f"-> Best parameters found: {grid_search.best_params_}")
    st.session_state.run_details['best_params'] = grid_search.best_params_

    # 6. Final Evaluation
    logs.append("\nStep 6: Final Evaluation")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logs.append(f"-> Final Model Accuracy: {accuracy:.2%}")
    
    st.session_state.run_details['logs'] = logs
    return best_model, accuracy, data, features, X_test, y_pred

# (Your existing generate_ai_summary and get_news_sentiment functions go here - no changes needed)
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
        model="llama3-70b-8192", 
    )
    return chat_completion.choices[0].message.content

# --- Streamlit App UI ---
st.title("AI Financial Analyst Dashboard")

# --- Sidebar ---
st.sidebar.header("User Input")
ticker_map = {
    "NVIDIA": "NVDA", "Apple": "AAPL", "Google": "GOOGL", 
    "Microsoft": "MSFT", "Amazon": "AMZN", "Tesla": "TSLA"
}
company_name = st.sidebar.selectbox("Select a Company", list(ticker_map.keys()))
ticker = ticker_map[company_name]

# Add link to the details page in the sidebar
st.sidebar.page_link("pages/1_Behind_the_Scenes.py", label="View Technical Details", icon="🛠️")


if ticker:
    # (Your existing UI code for the main page goes here - no changes needed)
    with st.spinner(f"Training advanced predictive model for {company_name}..."):
        model, accuracy, data, features, X_test, y_pred = train_prediction_model(ticker)
    
    st.header(f"Predictive Analysis for {company_name}")
    st.metric(label="Prediction Model Accuracy", value=f"{accuracy:.2%}")
    st.info("This model attempts to predict if the stock price will go up or down the next day. The interactive chart below shows its predictions on the most recent data.")

    st.subheader("Model Predictions vs. Actual Prices")
    predictions_df = X_test.copy()
    predictions_df['Actual_Close'] = data['Close'].loc[X_test.index]
    predictions_df['Prediction'] = y_pred
    plot_df = predictions_df.tail(100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual_Close'], mode='lines', name='Actual Price', line=dict(color='blue')))
    up_predictions = plot_df[plot_df['Prediction'] == 1]
    fig.add_trace(go.Scatter(x=up_predictions.index, y=up_predictions['Actual_Close'], mode='markers', name='Predicted Up', marker=dict(color='green', symbol='triangle-up', size=10)))
    down_predictions = plot_df[plot_df['Prediction'] == 0]
    fig.add_trace(go.Scatter(x=down_predictions.index, y=down_predictions['Actual_Close'], mode='markers', name='Predicted Down', marker=dict(color='red', symbol='triangle-down', size=10)))
    fig.update_layout(title=f'{company_name} - Actual Price vs. Model Predictions (Last 100 Days)', xaxis_title='Date', yaxis_title='Price (USD)', legend_title='Legend', height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    fig_importance, ax_importance = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax_importance)
    st.pyplot(fig_importance)

    st.header("AI-Generated Summary Report")
    if st.button("Generate AI Summary"):
        with st.spinner("Getting news sentiment and generating report..."):
            news_sentiment = get_news_sentiment(company_name)
            ai_summary = generate_ai_summary(ticker, company_name, accuracy, news_sentiment)
            st.markdown(ai_summary)