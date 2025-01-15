import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

def download_stock_data(ticker, years_of_history=5):
    """Download stock data with longer historical period for better long-term analysis"""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years_of_history*365)
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        return data
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators for risk assessment"""
    if data is None or data.empty:
        return None

    try:
        df = data.copy()

        # Moving averages
        df['SMA_200'] = df['Close'].rolling(window=min(200, len(df))).mean()
        df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()

        # Bollinger Bands (20-day, 2 standard deviations)
        window = min(20, len(df))
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()

        df['BB_middle'] = rolling_mean
        df['BB_upper'] = rolling_mean + (2 * rolling_std)
        df['BB_lower'] = rolling_mean - (2 * rolling_std)

        # Calculate returns and rolling metrics
        df['Returns'] = df['Close'].pct_change()

        # Use minimum of 252 days or available data length for rolling calculations
        window = min(252, len(df))
        df['Rolling_Vol'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        mean_returns = df['Returns'].rolling(window=window).mean()
        std_returns = df['Returns'].rolling(window=window).std()
        df['Rolling_Sharpe'] = np.where(std_returns != 0,
                                      mean_returns / std_returns * np.sqrt(252),
                                      0)

        return df
    except Exception as e:
        print(f"Error in technical indicators: {str(e)}")
        return None

def calculate_risk_metrics(data, risk_free_rate=0.05):
    """Calculate comprehensive risk metrics for long-term investment"""
    if data is None or data.empty:
        return None

    try:
        returns = data['Returns'].dropna()
        if len(returns) < 30:  # Minimum sample size for meaningful statistics
            raise ValueError("Insufficient data for risk metrics calculation")

        metrics = {}

        # Calculate basic metrics with error handling
        metrics['Annualized_Volatility'] = float(returns.std() * np.sqrt(252))

        # Sharpe Ratio
        excess_returns = returns.mean() - (risk_free_rate/252)
        if returns.std() != 0:
            metrics['Sharpe_Ratio'] = float(excess_returns / returns.std() * np.sqrt(252))
        else:
            metrics['Sharpe_Ratio'] = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            metrics['Sortino_Ratio'] = float(excess_returns / downside_returns.std() * np.sqrt(252))
        else:
            metrics['Sortino_Ratio'] = 0

        # Maximum Drawdown
        rolling_max = data['Close'].expanding(min_periods=1).max()
        drawdowns = data['Close'] / rolling_max - 1
        metrics['Max_Drawdown'] = float(drawdowns.min())

        # Other statistics
        metrics['Skewness'] = float(returns.skew()) if len(returns) > 30 else 0
        metrics['Kurtosis'] = float(returns.kurtosis()) if len(returns) > 30 else 0
        metrics['Value_at_Risk_95'] = float(np.percentile(returns, 5))
        metrics['Expected_Shortfall_95'] = float(returns[returns <= np.percentile(returns, 5)].mean())

        # Market Beta
        beta = calculate_beta(data['Returns'], '^NSEI')
        metrics['Beta'] = beta if beta is not None else 0

        return metrics

    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        return None

def calculate_beta(stock_returns, market_index):
    """Calculate beta against market index"""
    try:
        market_data = yf.download(market_index, start=stock_returns.index[0],
                                end=stock_returns.index[-1], progress=False)
        if market_data.empty:
            return None

        market_returns = market_data['Close'].pct_change()

        # Align dates
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()

        if len(aligned_data) < 30:
            return None

        cov_matrix = np.cov(aligned_data.iloc[:,0], aligned_data.iloc[:,1])
        if cov_matrix.size >= 4:  # Ensure we have a 2x2 covariance matrix
            beta = float(cov_matrix[0,1] / np.var(aligned_data.iloc[:,1]))
            return beta if not np.isnan(beta) else None
        return None
    except Exception as e:
        print(f"Error calculating beta: {str(e)}")
        return None

def format_percentage(value):
    """Safely format a number as a percentage string"""
    try:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%" if not np.isnan(value) else "N/A"
    except:
        return "N/A"

def format_decimal(value):
    """Safely format a number as a decimal string"""
    try:
        if value is None:
            return "N/A"
        return f"{value:.2f}" if not np.isnan(value) else "N/A"
    except:
        return "N/A"

def categorize_long_term_risk(metrics, market_cap):
    """Categorize investment risk considering multiple factors"""
    if metrics is None:
        return None

    try:
        market_cap_in_billions = float(market_cap) / 1e9 if market_cap else 0

        # Initialize score components
        size_score = 0
        volatility_score = 0
        quality_score = 0

        # Market Cap Scoring
        if market_cap_in_billions > 70:
            size_score = 3
        elif market_cap_in_billions > 20:
            size_score = 2
        else:
            size_score = 1

        # Volatility Scoring
        vol = metrics.get('Annualized_Volatility', float('inf'))
        if vol < 0.35:
            volatility_score = 3
        elif vol < 0.55:
            volatility_score = 2
        else:
            volatility_score = 1

        # Quality Scoring
        sharpe = metrics.get('Sharpe_Ratio', -float('inf'))
        max_dd = abs(metrics.get('Max_Drawdown', float('inf')))

        if sharpe > 1 and max_dd < 0.2:
            quality_score = 3
        elif sharpe > 0.5 and max_dd < 0.3:
            quality_score = 2
        else:
            quality_score = 1

        total_score = size_score + volatility_score + quality_score

        risk_category = "Very High Risk"
        if total_score >= 7:
            risk_category = "Low Risk"
        elif total_score >= 5:
            risk_category = "Moderate Risk"
        elif total_score >= 3:
            risk_category = "High Risk"

        analysis = {
            'risk_category': risk_category,
            'size_score': size_score,
            'volatility_score': volatility_score,
            'quality_score': quality_score,
            'total_score': total_score,
            'detailed_analysis': {
                'market_cap_analysis': f"Market Cap: ₹{format_decimal(market_cap_in_billions)}B",
                'volatility_analysis': f"Annualized Volatility: {format_percentage(metrics['Annualized_Volatility'])}",
                'quality_metrics': {
                    'sharpe_ratio': f"Sharpe Ratio: {format_decimal(metrics['Sharpe_Ratio'])}",
                    'sortino_ratio': f"Sortino Ratio: {format_decimal(metrics['Sortino_Ratio'])}",
                    'max_drawdown': f"Maximum Drawdown: {format_percentage(metrics['Max_Drawdown'])}",
                    'var_95': f"95% VaR: {format_percentage(metrics['Value_at_Risk_95'])}",
                    'beta': f"Beta: {format_decimal(metrics['Beta'])}"
                }
            }
        }

        return analysis
    except Exception as e:
        print(f"Error in risk categorization: {str(e)}")
        return None

def plot_risk_analysis(data):
    """Create comprehensive risk analysis plots"""
    if data is None or data.empty:
        return None

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Long-term Investment Risk Analysis', fontsize=16)

        # Price and Moving Averages
        axes[0,0].plot(data.index, data['Close'], label='Price')
        axes[0,0].plot(data.index, data['SMA_200'], label='200-day MA')
        axes[0,0].plot(data.index, data['SMA_50'], label='50-day MA')
        axes[0,0].set_title('Price Trends')
        axes[0,0].legend()

        # Volatility
        axes[0,1].plot(data.index, data['Rolling_Vol'], color='orange')
        axes[0,1].set_title('Rolling Annualized Volatility (252 days)')

        # Returns Distribution
        returns = data['Returns'].dropna()
        axes[1,0].hist(returns, bins=50, density=True, alpha=0.7)
        axes[1,0].set_title('Returns Distribution')

        # Rolling Sharpe Ratio
        axes[1,1].plot(data.index, data['Rolling_Sharpe'], color='green')
        axes[1,1].set_title('Rolling Sharpe Ratio (252 days)')

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        return None

def main(ticker="RELIANCE.NS"):
    try:
        # Download data
        print("Downloading data...")
        data = download_stock_data(ticker)
        if data is None:
            return None, None, None

        # Get stock info
        print("Fetching stock info...")
        stock_info = yf.Ticker(ticker)
        market_cap = stock_info.info.get('marketCap', 0)

        # Calculate indicators and metrics
        print("Calculating technical indicators...")
        data = calculate_technical_indicators(data)
        if data is None:
            return None, None, None

        print("Calculating risk metrics...")
        risk_metrics = calculate_risk_metrics(data)
        if risk_metrics is None:
            return data, None, None

        # Perform risk analysis
        print("Performing risk analysis...")
        risk_analysis = categorize_long_term_risk(risk_metrics, market_cap)
        if risk_analysis is None:
            return data, risk_metrics, None

        # Create visualization
        print("Creating visualizations...")
        fig = plot_risk_analysis(data)

        # Print results
        print("\n=== Long-term Investment Risk Analysis ===")
        print(f"\nRisk Category: {risk_analysis['risk_category']}")
        print("\nDetailed Analysis:")
        print(risk_analysis['detailed_analysis']['market_cap_analysis'])
        print(risk_analysis['detailed_analysis']['volatility_analysis'])
        print("\nQuality Metrics:")
        for metric, value in risk_analysis['detailed_analysis']['quality_metrics'].items():
            print(value)

        if fig is not None:
            plt.show()

        return data, risk_metrics, risk_analysis

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    data, metrics, analysis = main("RELIANCE.NS")