# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based stock analysis web application that provides:
- Real-time stock data retrieval using yfinance
- Technical indicators (MACD, RSI) with matplotlib charts
- Stock news aggregation 
- Interactive web dashboard with tabbed interface

## Architecture

**Backend (main.py:1-217)**
- FastAPI server with REST endpoints
- Stock data: `/stock/{symbol}/{period}` returns price history and company info
- Technical analysis: `/macd/{symbol}/{period}` and `/rsi/{symbol}/{period}` generate base64-encoded charts
- News: `/news/{symbol}` fetches and formats news articles
- Charts rendered server-side using matplotlib with Agg backend

**Frontend (static/index.html:1-449)**
- Single-page application using jQuery and DataTables
- Tab-based interface with lazy loading for charts and news
- Responsive design with grid layout for stock metrics
- Client-side data formatting and display

## Development Commands

**Install dependencies:**
```bash
uv sync
```

**Run development server:**
```bash
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Dependencies (pyproject.toml:7-17):**
- FastAPI/Uvicorn for web framework
- yfinance for stock data
- matplotlib/pandas/numpy for data processing
- ta library for technical indicators
- jinja2 for template support

## Key Implementation Details

- matplotlib configured with 'Agg' backend for server-side rendering
- Charts returned as base64-encoded PNG images
- Error handling with HTTPException for API endpoints
- News data structure parsing handles Yahoo Finance API format changes
- Static file serving for frontend assets