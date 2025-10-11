# app.py - FINAL Refactored Version

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

# --- Caching Functions (for pure, heavy computations) ---
@st.cache_data
def get_stock_data(ticker):
    """Downloads 2 years of hourly data."""
    return yf.Ticker(ticker).history(period="730d", interval="1h")

@st.cache_data
def get_historical_news(ticker, data):
    """Fetches historical news from Finnhub."""
    # (Your existing get_historical_news function code - no changes)
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

@st.cache_data
def engineer_features_and_split(data, news_df):
    """Performs all feature engineering and data splitting."""
    # (This function now contains the feature creation and splitting logic)
    if not news_df.empty:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment = sentiment_analyzer(news_df['headline'].tolist())
        news_df['sentiment_score'] = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiment]
        daily_sentiment = news_df.resample('D')['sentiment_score'].mean()
        daily_sentiment = daily_sentiment.reindex(data.index, method='ffill')
        data = data.join(daily_sentiment)
        data['sentiment_score'].fillna(0, inplace=True)
    else:
        data['sentiment_score'] = 0

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
    
    features = ['Close', 'Volume', 'sentiment_score', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'ATR']
    X = data[features]
    y = data['Target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    return data, X, features, X_train, X_test, y_train, y_test, tscv

@st.cache_data
def find_best_model(X_train, y_train, _tscv):
    """Performs hyperparameter tuning to find the best model."""
    # (This function is now dedicated to the heavy grid search task)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=_tscv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params

# --- Main Analysis Function (NOT cached) ---
def run_analysis(ticker):
    """Orchestrates the entire analysis and logs details to session_state."""
    if 'run_details' not in st.session_state:
        st.session_state.run_details = {}
    
    logs = []
    
    # Step 1 & 2: Data Collection
    data = get_stock_data(ticker)
    news_df = get_historical_news(ticker, data)
    logs.append(f"Step 1: Fetched {len(data)} hourly data points for {ticker}.")
    logs.append(f"Step 2: Fetched {len(news_df)} historical news headlines.")
    st.session_state.run_details['initial_data_head'] = data.head()

    # Step 3 & 4: Feature Engineering and Splitting
    data, X, features, X_train, X_test, y_train, y_test, tscv = engineer_features_and_split(data, news_df)
    logs.append("Step 3: Engineered features (Sentiment, Bollinger Bands, ATR, RSI).")
    st.session_state.run_details['featured_data_head'] = data.head()
    st.session_state.run_details['correlation_matrix'] = X.corr()
    logs.append(f"Step 4: Performed TimeSeriesSplit. Training data shape: {X_train.shape}. Testing data shape: {X_test.shape}.")

    # Step 5: Hyperparameter Tuning
    logs.append("Step 5: Searching for optimal model with GridSearchCV...")
    best_model, best_params = find_best_model(X_train, y_train, tscv)
    logs.append(f"-> Best parameters found: {best_params}")
    st.session_state.run_details['best_params'] = best_params

    # Step 6: Final Evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logs.append(f"Step 6: Final Model Accuracy on test data: {accuracy:.2%}")
    
    # Store everything in session state
    st.session_state.run_details['logs'] = logs
    st.session_state.run_details['model'] = best_model
    st.session_state.run_details['accuracy'] = accuracy
    st.session_state.run_details['data'] = data
    st.session_state.run_details['features'] = features
    st.session_state.run_details['X_test'] = X_test
    st.session_state.run_details['y_pred'] = y_pred

# (Your existing get_news_sentiment and generate_ai_summary functions go here)
@st.cache_data
def get_news_sentiment(company_name):
    # (No changes needed in this function)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = f"https://www.google.com/search?q={company_name}+stock+news&tbm=nws"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [item.get_text() for item in soup.find_all('div', {'class': 'n0jPhd ynAwRc MBeuO nDgy9d'}, limit=5)]
    if not headlines: return "Could not retrieve recent news headlines."
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = sentiment_analyzer(headlines)
    analysis = f"Recent News Sentiment for {company_name}:\n"
    for i, headline in enumerate(headlines):
        analysis += f"- Headline: {headline}\n  - Sentiment: {results[i]['label']} (Score: {results[i]['score']:.2f})\n"
    return analysis

def generate_ai_summary(ticker, company_name, model_accuracy, news_sentiment):
    # (No changes needed in this function)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    system_prompt = "You are a senior financial analyst AI..."
    human_prompt = f"Please generate a report for {company_name} ({ticker}) based on...\n--- PREDICTIVE MODEL ACCURACY ---\n{model_accuracy:.2%}\n--- NEWS SENTIMENT DATA ---\n{news_sentiment}"
    chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": human_prompt}], model="llama-3.3-70b-versatile")
    return chat_completion.choices[0].message.content

# --- Streamlit App UI ---
st.title("AI Financial Analyst Dashboard")

st.sidebar.header("User Input")
ticker_map = {"NVIDIA": "NVDA", "Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Amazon": "AMZN", "Tesla": "TSLA"}
company_name = st.sidebar.selectbox("Select a Company", list(ticker_map.keys()))
ticker = ticker_map[company_name]
st.sidebar.page_link("pages/1_Behind_the_Scenes.py", label="View Technical Details", icon="🛠️")

# --- Main Logic ---
# Run the analysis only once when the ticker changes
if 'current_ticker' not in st.session_state or st.session_state.current_ticker != ticker:
    with st.spinner(f"Running full analysis for {company_name}... This may take a minute."):
        run_analysis(ticker)
        st.session_state.current_ticker = ticker

# --- Display Results from Session State ---
details = st.session_state.run_details
if details:
    st.header(f"Predictive Analysis for {company_name}")
    st.metric(label="Prediction Model Accuracy", value=f"{details['accuracy']:.2%}")
    st.info("Interactive chart showing model predictions on the most recent hourly data.")

    predictions_df = details['X_test'].copy()
    predictions_df['Actual_Close'] = details['data']['Close'].loc[details['X_test'].index]
    predictions_df['Prediction'] = details['y_pred']
    plot_df = predictions_df.tail(168)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual_Close'], mode='lines', name='Actual Price'))
    up = plot_df[plot_df['Prediction'] == 1]
    down = plot_df[plot_df['Prediction'] == 0]
    fig.add_trace(go.Scatter(x=up.index, y=up['Actual_Close'], mode='markers', name='Predicted Up', marker=dict(color='green', symbol='triangle-up', size=8)))
    fig.add_trace(go.Scatter(x=down.index, y=down['Actual_Close'], mode='markers', name='Predicted Down', marker=dict(color='red', symbol='triangle-down', size=8)))
    fig.update_layout(title=f'{company_name} - Actual Price vs. Predictions (Last 168 Hours)', height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    feature_importances = pd.Series(details['model'].feature_importances_, index=details['features']).sort_values(ascending=False)
    fig_importance, ax_importance = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax_importance)
    st.pyplot(fig_importance)

    st.header("AI-Generated Summary Report")
    if st.button("Generate AI Summary"):
        with st.spinner("Getting recent news and generating report..."):
            news_sentiment = get_news_sentiment(company_name)
            ai_summary = generate_ai_summary(ticker, company_name, details['accuracy'], news_sentiment)
            st.markdown(ai_summary)
else:
    st.info("Select a company from the sidebar to begin analysis.")