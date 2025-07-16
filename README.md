# Stock Analysis Dashboard

A FastAPI-based web application for real-time stock analysis, featuring technical indicators, price charts, and news aggregation.

## Features

- **Real-time Stock Data**: Retrieve current and historical stock prices using Yahoo Finance
- **Technical Analysis**: Generate MACD and RSI indicators with interactive charts
- **News Aggregation**: Fetch and display relevant stock news articles with thumbnails
- **Stock Comparison**: Compare multiple stocks side-by-side with normalized performance charts
- **Correlation Analysis**: Analyze correlation between multiple stocks with heatmap visualization
- **Portfolio Analysis**: Evaluate portfolio performance with risk metrics (Sharpe ratio, volatility)
- **Volatility Analysis**: Compare risk levels across stocks with rolling volatility charts
- **Interactive Dashboard**: Web-based interface with tabbed navigation and lazy loading
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Python 3.8+
- UV package manager (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gavi/stocks
cd stocks
```

2. Install dependencies:
```bash
uv sync
```

### Running the Application

Start the development server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`

## API Endpoints

### Single Stock Analysis
- `GET /stock/{symbol}/{period}` - Get stock price history and company info
- `GET /macd/{symbol}/{period}` - Generate MACD technical indicator chart
- `GET /rsi/{symbol}/{period}` - Generate RSI technical indicator chart
- `GET /news/{symbol}` - Fetch recent news articles for a stock

### Multi-Stock Analysis
- `GET /compare/{symbols}/{period}` - Compare multiple stocks (comma-separated symbols)
- `GET /compare/chart/{symbols}/{period}` - Generate comparison chart with normalized performance
- `GET /correlation/{symbols}/{period}` - Analyze correlation between stocks with heatmap
- `GET /portfolio/{symbols}/{period}` - Portfolio analysis with risk metrics
- `GET /volatility/{symbols}/{period}` - Volatility analysis and comparison

### Parameters

- `symbol`: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
- `symbols`: Comma-separated stock symbols for multi-stock analysis (e.g., AAPL,MSFT,GOOGL)
- `period`: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

## Technology Stack

- **Backend**: FastAPI, Python
- **Data Source**: Yahoo Finance (yfinance)
- **Charts**: Matplotlib with server-side rendering
- **Technical Analysis**: TA-Lib library
- **Data Processing**: Pandas, NumPy
- **Visualization**: Seaborn for correlation heatmaps
- **Frontend**: HTML, CSS, JavaScript (jQuery, DataTables)

## Project Structure

```
stocks/
├── main.py           # FastAPI application and API endpoints
├── static/
│   └── index.html    # Frontend dashboard
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

## How to Use

### Single Stock Analysis
Enter a single stock symbol (e.g., `AAPL`) to analyze:
- View stock data and price history
- Generate MACD and RSI technical indicators
- Read latest news articles

### Multi-Stock Analysis
Enter multiple stock symbols separated by commas (e.g., `AAPL,MSFT,GOOGL`) to:
- Compare stock performance side-by-side
- Analyze correlation between stocks
- Evaluate portfolio metrics (Sharpe ratio, volatility)
- Compare risk levels across stocks

## Development

The application uses:
- FastAPI for the REST API backend
- Matplotlib with 'Agg' backend for server-side chart generation
- Base64-encoded PNG images for chart delivery
- jQuery and DataTables for frontend interactivity
- Seaborn for advanced statistical visualizations

## License

MIT License