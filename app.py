import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os
from groq import Groq
from google.colab import userdata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_quantitative_analysis(ticker_symbol):
    """
    Fetches historical stock data and calculates key technical indicators.
    """
    print(f"🔬 Performing quantitative analysis for {ticker_symbol}...")
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="3mo") # Get 3 months of data

        if hist.empty:
            return "Could not retrieve historical data. The ticker might be invalid."

        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        # Calculate MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        last_macd = macd.iloc[-1]
        last_signal = signal_line.iloc[-1]

        # Create a summary string
        latest_price = hist['Close'].iloc[-1]
        analysis = f"""
        Quantitative Analysis for {ticker_symbol}:
        - Latest Closing Price: ${latest_price:.2f}
        - Relative Strength Index (RSI): {last_rsi:.2f} (A value > 70 may indicate overbought, < 30 may indicate oversold)
        - MACD: {last_macd:.2f}
        - MACD Signal Line: {last_signal:.2f} (A bullish signal can occur when MACD crosses above the signal line)
        """
        return analysis
    except Exception as e:
        return f"An error occurred during quantitative analysis: {e}"
    
def get_news_sentiment_analysis(company_name):
    """
    Scrapes Google News for headlines and performs sentiment analysis.
    """
    print(f"📰 Performing news sentiment analysis for {company_name}...")
    try:
        # We use a special User-Agent to pretend we're a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = f"https://www.google.com/search?q={company_name}+stock+news&tbm=nws"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = []
        for item in soup.find_all('div', {'class': 'n0jPhd ynAwRc MBeuO nDgy9d'}, limit=5):
            headlines.append(item.get_text())

        if not headlines:
            return "Could not retrieve any news headlines."

        # Use a pre-trained NLP model for sentiment analysis
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        results = sentiment_analyzer(headlines)

        # Format the results
        analysis = f"\nNews Sentiment Analysis for {company_name}:\n"
        for i, headline in enumerate(headlines):
            sentiment = results[i]['label']
            score = results[i]['score']
            analysis += f"- Headline: {headline}\n  - Sentiment: {sentiment} (Score: {score:.2f})\n"

        return analysis
    except Exception as e:
        return f"An error occurred during news analysis: {e}"
    
def run_analyst_agent(ticker, company_name):
    """
    The main agent function that orchestrates the analysis and generates a report.
    """
    print(f"\n🚀 Starting Analysis for {company_name} ({ticker})...")

    # Step 1: Run the tools to gather data
    quant_results = get_quantitative_analysis(ticker)
    news_results = get_news_sentiment_analysis(company_name)

    print("\n✅ All data gathered. Now generating the final report with AI...")

    # Step 2: Use the LLM to synthesize the results into a report
    try:
        client = Groq(api_key=userdata.get('GROQ_API_KEY'))

        system_prompt = """
        You are a senior financial analyst AI. Your task is to write a concise, easy-to-understand investment briefing.
        You will be given a quantitative analysis and a news sentiment analysis.
        Synthesize this information into a final report with three sections:
        1. **Summary:** A brief, one-paragraph overview of the situation.
        2. **Quantitative Outlook:** Explain the key metrics (RSI, MACD) in simple terms and what they suggest.
        3. **News & Sentiment Outlook:** Summarize the sentiment of recent news and what it implies.
        Do not include any investment advice. Just present the facts and analysis clearly.
        """

        human_prompt = f"""
        Please generate a report for {company_name} ({ticker}) based on the following data:

        --- QUANTITATIVE DATA ---
        {quant_results}

        --- NEWS SENTIMENT DATA ---
        {news_results}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model="llama-3.3-70b-versatile", # <-- THIS IS THE CORRECTED LINE
        )
        final_report = chat_completion.choices[0].message.content
        print("\n" + "="*50)
        print("📈 FINAL INVESTMENT BRIEFING 📈")
        print("="*50)
        print(final_report)

    except Exception as e:
        print(f"An error occurred while generating the final report: {e}")

# --- RUN THE AGENT ---
run_analyst_agent(ticker="NVDA", company_name="NVIDIA")

def create_feature_rich_dataset(ticker_symbol):
    """
    Downloads 2 years of stock data and engineers a rich set of features.
    """
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="2y")

    # Standard technical indicators
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

    # Volatility
    hist['Volatility'] = hist['Close'].rolling(window=20).std() * np.sqrt(20)

    # Momentum (RSI)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # Create the target variable: 1 if next day's price went up, 0 if it went down
    hist['Target'] = (hist['Close'].shift(-1) > hist['Close']).astype(int)

    # Drop rows with missing values created by rolling windows and the final row
    hist.dropna(inplace=True)

    return hist

# --- Create the dataset ---
ticker = "NVDA"
data = create_feature_rich_dataset(ticker)

# Display the first few rows of your new dataset
print(f"Feature-rich dataset for {ticker}:")
data.head()

print("--- Starting Exploratory Data Analysis (EDA) ---")

# 1. Plot the closing price over time
plt.figure(figsize=(14, 7))
plt.title(f'{ticker} Closing Price Over Time')
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA_50'], label='50-Day SMA', linestyle='--')
plt.plot(data['SMA_200'], label='200-Day SMA', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

# 2. Create the correlation matrix
# We select only the numeric columns for correlation calculation
numeric_cols = data.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()

# 3. Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f'Correlation Matrix of Features for {ticker}')
plt.show()

print("\n--- EDA Complete ---")
print("The heatmap above shows the correlation between different features.")
print("Values close to 1 or -1 indicate a strong relationship.")
print("Look at the 'Target' row to see which features are most correlated with the price movement.")

print("\n--- Starting Model Training ---")

# Define features (X) and target (y)
features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_20', 'Volatility', 'RSI']
X = data[features]
y = data['Target']

# Time-series split (train on older data, test on newer data)
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("\n--- Model Training Complete ---")