from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Import your existing Python functions
from rough import (
    download_stock_data,
    calculate_technical_indicators,
    calculate_risk_metrics,
    categorize_long_term_risk,
    plot_risk_analysis
)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Stock Risk Analysis API"}

@app.get("/analyze/{ticker}")
def analyze_stock(ticker: str):
    try:
        # Fetch data and perform analysis
        data = download_stock_data(ticker)
        if data is None:
            return JSONResponse({"error": "Failed to fetch stock data"}, status_code=400)

        data = calculate_technical_indicators(data)
        risk_metrics = calculate_risk_metrics(data)
        stock_info = yf.Ticker(ticker).info
        market_cap = stock_info.get("marketCap", 0)
        risk_analysis = categorize_long_term_risk(risk_metrics, market_cap)

        # Convert plot to base64
        fig = plot_risk_analysis(data)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        img_buffer.close()

        return {
            "metrics": risk_metrics,
            "analysis": risk_analysis,
            "plot": img_base64,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
