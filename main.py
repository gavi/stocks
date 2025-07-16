import io
import base64
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Stock Analysis API", description="API for stock analysis with technical indicators")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/stock/{symbol}/{period}")
async def get_stock_data(symbol: str, period: str = "3mo"):
    try:
        ticker = yf.Ticker(symbol.upper())
        
        info = ticker.info
        history = ticker.history(period=period)
        
        if history.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        history_data = []
        for date, row in history.iterrows():
            history_data.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Open": float(row['Open']),
                "High": float(row['High']),
                "Low": float(row['Low']),
                "Close": float(row['Close']),
                "Volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol.upper(),
            "info": info,
            "history": history_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare/{symbols}/{period}")
async def compare_stocks(symbols: str, period: str = "3mo"):
    """Compare multiple stocks side by side. Symbols should be comma-separated."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for comparison")
        
        if len(symbol_list) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 symbols allowed for comparison")
        
        comparison_data = []
        
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period=period)
                
                if history.empty:
                    continue
                
                # Calculate percentage change from first day for normalization
                first_close = history['Close'].iloc[0]
                history['Normalized'] = ((history['Close'] - first_close) / first_close * 100)
                
                # Get latest metrics
                latest = history.iloc[-1]
                
                comparison_data.append({
                    "symbol": symbol,
                    "name": info.get('longName', symbol),
                    "current_price": float(latest['Close']),
                    "change_percent": float(history['Normalized'].iloc[-1]),
                    "volume": int(latest['Volume']),
                    "market_cap": info.get('marketCap'),
                    "pe_ratio": info.get('forwardPE'),
                    "history": [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "close": float(row['Close']),
                            "normalized": float(row['Normalized'])
                        }
                        for date, row in history.iterrows()
                    ]
                })
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not comparison_data:
            raise HTTPException(status_code=404, detail="No valid stock data found")
        
        return {
            "symbols": symbol_list,
            "period": period,
            "stocks": comparison_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare/chart/{symbols}/{period}")
async def get_comparison_chart(symbols: str, period: str = "3mo"):
    """Generate comparison chart for multiple stocks."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for comparison")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, symbol in enumerate(symbol_list):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    continue
                
                # Normalize to percentage change from first day
                first_close = data['Close'].iloc[0]
                normalized = ((data['Close'] - first_close) / first_close * 100)
                
                color = colors[i % len(colors)]
                
                # Price chart (normalized)
                ax1.plot(data.index, normalized, label=f'{symbol}', linewidth=2, color=color)
                
                # Volume chart
                ax2.bar(data.index, data['Volume'], alpha=0.6, label=f'{symbol} Volume', color=color)
                
            except Exception as e:
                print(f"Error plotting {symbol}: {e}")
                continue
        
        ax1.set_title('Stock Price Comparison (Normalized % Change)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Percentage Change (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('Volume Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/macd/{symbol}/{period}")
async def get_macd_chart(symbol: str, period: str = "3mo"):
    try:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        macd_line = ta.trend.MACD(data['Close']).macd()
        macd_signal = ta.trend.MACD(data['Close']).macd_signal()
        macd_histogram = ta.trend.MACD(data['Close']).macd_diff()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        ax1.set_title(f'{symbol.upper()} - Stock Price')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data.index, macd_line, label='MACD Line', linewidth=2)
        ax2.plot(data.index, macd_signal, label='Signal Line', linewidth=2)
        ax2.bar(data.index, macd_histogram, label='MACD Histogram', alpha=0.6)
        ax2.set_title('MACD Indicator')
        ax2.set_ylabel('MACD')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rsi/{symbol}/{period}")
async def get_rsi_chart(symbol: str, period: str = "3mo"):
    try:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        ax1.set_title(f'{symbol.upper()} - Stock Price')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data.index, rsi, label='RSI', linewidth=2, color='orange')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart": img_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/{symbol}")
async def get_stock_news(symbol: str):
    try:
        ticker = yf.Ticker(symbol.upper())
        news = ticker.news
        
        if not news:
            return {"news": [], "symbol": symbol.upper()}
        
        formatted_news = []
        for article_wrapper in news[:10]:  # Limit to 10 most recent articles
            # Extract the content from the wrapper
            article = article_wrapper.get("content", {})
            
            if not article:
                continue
            
            # Extract data from the correct structure
            title = article.get("title", "")
            
            # Get URL from clickThroughUrl or canonicalUrl
            link = ""
            if article.get("clickThroughUrl"):
                link = article["clickThroughUrl"].get("url", "")
            elif article.get("canonicalUrl"):
                link = article["canonicalUrl"].get("url", "")
            
            # Get publisher from provider
            publisher = ""
            if article.get("provider"):
                publisher = article["provider"].get("displayName", "")
            
            # Convert pubDate to timestamp
            publish_time = 0
            if article.get("pubDate"):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(article["pubDate"].replace('Z', '+00:00'))
                    publish_time = int(dt.timestamp())
                except:
                    pass
            
            # Handle thumbnail
            thumbnail = ""
            if article.get("thumbnail") and article["thumbnail"].get("resolutions"):
                # Use the smallest resolution for thumbnails
                resolutions = article["thumbnail"]["resolutions"]
                if resolutions:
                    # Find the smallest resolution (likely the last one)
                    for res in reversed(resolutions):
                        if res.get("url"):
                            thumbnail = res["url"]
                            break
            
            summary = article.get("summary", "") or article.get("description", "")
            
            formatted_article = {
                "title": title,
                "link": link,
                "publisher": publisher,
                "publishTime": publish_time,
                "thumbnail": thumbnail,
                "summary": summary
            }
            formatted_news.append(formatted_article)
        
        return {
            "news": formatted_news,
            "symbol": symbol.upper()
        }
    except Exception as e:
        print(f"Error getting news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/correlation/{symbols}/{period}")
async def get_correlation_analysis(symbols: str, period: str = "3mo"):
    """Analyze correlation between multiple stocks."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for correlation analysis")
        
        # Get stock data
        stock_data = {}
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period)
                if not history.empty:
                    stock_data[symbol] = history['Close']
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        if len(stock_data) < 2:
            raise HTTPException(status_code=404, detail="Not enough valid stock data for correlation analysis")
        
        # Create DataFrame with all stocks
        df = pd.DataFrame(stock_data)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Stock Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "chart": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{symbols}/{period}")
async def get_portfolio_analysis(symbols: str, period: str = "3mo"):
    """Analyze portfolio performance and risk metrics."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for portfolio analysis")
        
        # Get stock data
        stock_data = {}
        stock_returns = {}
        
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period)
                if not history.empty:
                    stock_data[symbol] = history['Close']
                    stock_returns[symbol] = history['Close'].pct_change().dropna()
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        if len(stock_data) < 2:
            raise HTTPException(status_code=404, detail="Not enough valid stock data for portfolio analysis")
        
        # Create equal-weight portfolio (you could make this configurable)
        weights = np.array([1/len(stock_data)] * len(stock_data))
        
        # Calculate portfolio metrics
        returns_df = pd.DataFrame(stock_returns)
        returns_df = returns_df.dropna()
        
        # Portfolio return
        portfolio_returns = (returns_df * weights).sum(axis=1)
        total_return = ((1 + portfolio_returns).cumprod().iloc[-1] - 1) * 100
        
        # Portfolio volatility (annualized)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_return / (portfolio_returns.std() * np.sqrt(252))
        
        # Best and worst performers
        individual_returns = {}
        for symbol in stock_data.keys():
            individual_returns[symbol] = ((stock_data[symbol].iloc[-1] / stock_data[symbol].iloc[0]) - 1) * 100
        
        best_performer = max(individual_returns.items(), key=lambda x: x[1])
        worst_performer = min(individual_returns.items(), key=lambda x: x[1])
        
        # Create performance chart
        plt.figure(figsize=(12, 8))
        
        # Plot individual stocks
        for symbol, prices in stock_data.items():
            normalized_prices = (prices / prices.iloc[0]) * 100
            plt.plot(prices.index, normalized_prices, label=f'{symbol}', linewidth=2, alpha=0.7)
        
        # Plot portfolio
        portfolio_df = pd.DataFrame(stock_data)
        portfolio_normalized = (portfolio_df / portfolio_df.iloc[0]) * 100
        portfolio_performance = (portfolio_normalized * weights).sum(axis=1)
        plt.plot(portfolio_df.index, portfolio_performance, label='Equal-Weight Portfolio', 
                linewidth=3, color='black', linestyle='--')
        
        plt.title('Portfolio vs Individual Stock Performance', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Base = 100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "portfolio_metrics": {
                "total_return": float(total_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "best_performer": {
                    "symbol": best_performer[0],
                    "return": float(best_performer[1])
                },
                "worst_performer": {
                    "symbol": worst_performer[0],
                    "return": float(worst_performer[1])
                }
            },
            "period": period,
            "chart": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/volatility/{symbols}/{period}")
async def get_volatility_analysis(symbols: str, period: str = "3mo"):
    """Analyze volatility comparison between stocks."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for volatility analysis")
        
        # Get stock data and calculate volatility
        volatility_data = []
        stock_returns = {}
        
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period)
                if not history.empty:
                    returns = history['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    
                    volatility_data.append({
                        "symbol": symbol,
                        "volatility": float(volatility)
                    })
                    stock_returns[symbol] = returns
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        if len(volatility_data) < 2:
            raise HTTPException(status_code=404, detail="Not enough valid stock data for volatility analysis")
        
        # Create volatility comparison chart
        plt.figure(figsize=(12, 8))
        
        # Bar chart of volatilities
        symbols = [item['symbol'] for item in volatility_data]
        volatilities = [item['volatility'] for item in volatility_data]
        
        colors = ['#dc3545' if v > 30 else '#ffc107' if v > 20 else '#28a745' for v in volatilities]
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(symbols, volatilities, color=colors, alpha=0.7)
        plt.title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Volatility (%)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, vol in zip(bars, volatilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rolling volatility over time
        plt.subplot(2, 1, 2)
        for symbol in stock_returns.keys():
            rolling_vol = stock_returns[symbol].rolling(window=20).std() * np.sqrt(252) * 100
            plt.plot(rolling_vol.index, rolling_vol, label=f'{symbol}', linewidth=2)
        
        plt.title('Rolling 20-Day Volatility', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "volatility_metrics": volatility_data,
            "chart": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)