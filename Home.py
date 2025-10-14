# Home.py - FINAL Corrected Version

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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst Dashboard", page_icon="🤖", layout="wide")

# Home.py

# --- Add these new imports at the top of your file ---
import time
import io

@st.cache_data
def get_stock_data(ticker):
    """
    Downloads 2 years of historical hourly stock data from Alpha Vantage.
    This is a complex function to handle the API's monthly data slices.
    """
    st.write(f"Fetching 2 years of hourly data for {ticker} from Alpha Vantage...")
    
    API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    all_data = []

    # Alpha Vantage provides 2 years of data in 24 monthly slices
    for year in [1, 2]:
        for month in range(1, 13):
            slice_period = f'year{year}month{month}'
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=60min&slice={slice_period}&apikey={API_KEY}'
            
            try:
                response = requests.get(url)
                response.raise_for_status() # Raise an exception for bad status codes
                
                # The data is returned as CSV, so we read it into a DataFrame
                df = pd.read_csv(io.StringIO(response.text))
                all_data.append(df)
                
                # --- Crucial: Add a delay to respect the API's rate limit ---
                time.sleep(13) # Sleep for 13 seconds between each of the 24 calls
            
            except requests.exceptions.RequestException as e:
                st.warning(f"Could not fetch data for slice {slice_period}. Error: {e}. Skipping.")
                continue
            except Exception as e:
                # Handle cases where the CSV might be empty or malformed for a given month
                st.info(f"No data available for {ticker} in period {slice_period}. This may be normal.")
                time.sleep(13)
                continue

    if not all_data:
        st.error(f"Failed to fetch any data for {ticker} from Alpha Vantage. The ticker might be invalid or the API limit may have been reached.")
        return pd.DataFrame()

    # Combine all the monthly dataframes into one
    full_df = pd.concat(all_data)
    
    # --- Data Cleaning and Formatting ---
    # Convert 'time' column to datetime objects and set as index
    full_df.set_index(pd.to_datetime(full_df['time']), inplace=True)
    
    # Rename columns to match our existing code
    full_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    # Select only the columns we need
    full_df = full_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Sort the data chronologically
    full_df.sort_index(inplace=True)
    
    # Convert index to America/New_York timezone to match news data
    full_df.index = full_df.index.tz_localize('UTC').tz_convert('America/New_York')

    return full_df
@st.cache_data
def get_historical_news(ticker, data):
    """Fetches historical news from Finnhub for the date range of the stock data."""
    
    # --- FIX: Add a check to ensure the data is not empty ---
    if data.empty:
        st.warning(f"Could not retrieve historical stock data for {ticker}. Skipping news analysis.")
        return pd.DataFrame() # Return an empty DataFrame to prevent a crash
    # --- END OF FIX ---

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

@st.cache_data
def get_news_sentiment(company_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
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

@st.cache_data
def engineer_features_and_split(data, news_df):
    """Performs all feature engineering and data splitting."""
    if not news_df.empty:
        # (Your sentiment analysis logic is here)
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        sentiment = sentiment_analyzer(news_df['headline'].tolist())
        news_df['sentiment_score'] = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiment]
        daily_sentiment = news_df.resample('D')['sentiment_score'].mean().reindex(data.index, method='ffill')
        data = data.join(daily_sentiment)
        data['sentiment_score'] = data['sentiment_score'].fillna(0)
    else:
        data['sentiment_score'] = 0

    # (Your feature engineering logic is here)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Std_Dev_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['Std_Dev_20'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['Std_Dev_20'] * 2)
    high_low, high_close, low_close = data['High'] - data['Low'], np.abs(data['High'] - data['Close'].shift()), np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    data['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + ((data['Close'].diff().where(data['Close'].diff() > 0, 0)).rolling(14).mean() / (-data['Close'].diff().where(data['Close'].diff() < 0, 0)).rolling(14).mean())))
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    data.dropna(inplace=True)

    # --- FIX: Add a check to ensure data exists after processing ---
    if data.empty:
        st.error(f"Insufficient historical data available for the selected stock to train a model. Please select another one.")
        st.stop() # This halts the script gracefully
    # --- END OF FIX ---

    features = ['Close', 'Volume', 'sentiment_score', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'ATR']
    X, y = data[features], data['Target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        
    return data, X, features, X_train, X_test, y_train, y_test, tscv

@st.cache_data
def find_best_model(X_train, y_train, _tscv):
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=_tscv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def generate_ai_summary(ticker, company_name, model_accuracy, news_sentiment):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    system_prompt = "You are a senior financial analyst AI..."
    human_prompt = f"Report for {company_name} ({ticker})...\n--- ACCURACY ---\n{model_accuracy:.2%}\n--- SENTIMENT ---\n{news_sentiment}"
    chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": human_prompt}], model="llama-3.3-70b-versatile")
    return chat_completion.choices[0].message.content

# --- Main Analysis Function ---
# Home.py

# --- Main Analysis Function ---
def run_analysis(ticker):
    """Orchestrates the analysis and logs details to session_state."""
    logs = []
    
    # --- FIX: Add a data validation check right after fetching ---
    data = get_stock_data(ticker)
    if data.empty:
        st.error(f"Could not retrieve historical stock data for {ticker}. The ticker may be invalid or delisted, or there may be an issue with the data provider. Please select another stock.")
        st.stop() # This halts the script gracefully
    # --- END OF FIX ---

    news_df = get_historical_news(ticker, data)
    logs.append(f"## Step 1 & 2: Data Collection\n- Fetched **{len(data)}** hourly data points and **{len(news_df)}** news headlines.")
    
    data, X, features, X_train, X_test, y_train, y_test, tscv = engineer_features_and_split(data, news_df)
    
    # Check if data is empty after feature engineering
    if X_train.empty or X_test.empty:
        st.error(f"Insufficient data for {ticker} after processing to train a model. Please select another stock.")
        st.stop()

    logs.append(f"\n## Step 3 & 4: Feature Engineering & Splitting\n- **Splitting Method:** `TimeSeriesSplit` to train on the past and test on the future.\n- Final training data shape: `{X_train.shape}`. Testing data shape: `{X_test.shape}`.")
    
    best_model, best_params = find_best_model(X_train, y_train, tscv)
    logs.append(f"\n## Step 5: Model Selection & Tuning\n- **Algorithm:** `XGBoost` for its high performance.\n- Best parameters found by `GridSearchCV`: `{best_params}`")
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logs.append(f"\n## Step 6: Final Evaluation\n- Final Model Accuracy on test data: **{accuracy:.2%}**")
    
    st.session_state.run_details = {
        'logs': logs, 'model': best_model, 'accuracy': accuracy, 'data': data, 
        'features': features, 'X_test': X_test, 'y_pred': y_pred, 
        'correlation_matrix': X.corr(), 'best_params': best_params, 
        'initial_data_head': get_stock_data(ticker).head(), 
        'featured_data_head': data.head()
    }

# --- Streamlit App UI ---
st.title("AI Financial Analyst Dashboard")
st.sidebar.header("User Input")
ticker_map = {"NVIDIA": "NVDA", "Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Amazon": "AMZN", "Tesla": "TSLA", "Meta Platforms": "META", "JPMorgan Chase": "JPM", "Visa": "V", "Mastercard": "MA", "Johnson & Johnson": "JNJ", "Walmart": "WMT", "Procter & Gamble": "PG", "Home Depot": "HD", "Salesforce": "CRM", "Adobe": "ADBE", "Intel": "INTC", "Cisco Systems": "CSCO", "Oracle": "ORCL", "Accenture": "ACN", "Netflix": "NFLX", "Goldman Sachs": "GS", "Morgan Stanley": "MS", "American Express": "AXP"}
company_name = st.sidebar.selectbox("Select a Company", list(ticker_map.keys()))
ticker = ticker_map[company_name]

if 'current_ticker' not in st.session_state or st.session_state.current_ticker != ticker:
    with st.spinner(f"Running full analysis for {company_name}..."):
        run_analysis(ticker)
        st.session_state.current_ticker = ticker

details = st.session_state.run_details
if details:
    st.header(f"Predictive Analysis for {company_name}")
    st.metric(label="Prediction Model Accuracy", value=f"{details['accuracy']:.2%}")

        # --- Add This New Code Block ---
    with st.expander("💡 How to Interpret this Accuracy Score"):
        st.markdown("""
            Predicting hourly stock movements is one of the most challenging problems in data science due to the market's high degree of randomness.
            
            - **50% Accuracy:** Is equivalent to a random coin flip.
            - **55% - 60% Accuracy:** Is considered a significant and valuable edge. Top-tier quantitative hedge funds often build complex strategies around models with this level of accuracy.
            - **Above 65% Accuracy:** Consistently achieving this level is extremely rare and would represent a world-class breakthrough.
            
            Our model's performance should be viewed in this context. An accuracy consistently above 50% indicates that it has successfully identified a genuine, predictive signal from the noise.
        """)
    # --- End of New Code Block ---

    # --- UPDATED: Plotly chart with full test data ---
    predictions_df = details['X_test'].copy()
    predictions_df['Actual_Close'] = details['data']['Close'].loc[details['X_test'].index]
    predictions_df['Prediction'] = details['y_pred']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Actual_Close'], mode='lines', name='Actual Price'))
    up = predictions_df[predictions_df['Prediction'] == 1]
    down = predictions_df[predictions_df['Prediction'] == 0]
    fig.add_trace(go.Scatter(x=up.index, y=up['Actual_Close'], mode='markers', name='Predicted Up', marker=dict(color='green', symbol='triangle-up', size=8)))
    fig.add_trace(go.Scatter(x=down.index, y=down['Actual_Close'], mode='markers', name='Predicted Down', marker=dict(color='red', symbol='triangle-down', size=8)))
    fig.update_layout(title=f'{company_name} - Actual Price vs. Predictions (Full Test Period)', height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.header("AI-Generated Summary Report")
    if st.button("Generate AI Summary"):
        with st.spinner("Getting recent news and generating report..."):
            news_sentiment = get_news_sentiment(company_name)
            ai_summary = generate_ai_summary(ticker, company_name, details['accuracy'], news_sentiment)
            st.markdown(ai_summary)