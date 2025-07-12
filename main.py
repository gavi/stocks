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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)