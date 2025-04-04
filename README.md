# CT Lighthouse Lite - Trading Performance Dashboard
[![Docker](https://github.com/ItsNewe/ct-lighthouse-lite/actions/workflows/docker-image.yml/badge.svg)](https://github.com/ItsNewe/ct-lighthouse-lite/actions/workflows/docker-image.yml)

CT Lighthouse Lite is a Streamlit-based dashboard for detailed analysis of cTrader algorithmic trading backtest results. This tool helps traders visualize and understand their trading performance through comprehensive metrics.

## Features

- **Performance Overview**: Get key metrics including win rate, profit factor, Sharpe ratio, and drawdown analysis
- **Time-Based Analysis**: Analyze performance by weekday and hour to identify optimal trading periods
- **Position Analysis**: Compare performance between different trade types (Buy/Sell)
- **Risk Management**: Track performance against risk parameters like max daily loss and max total loss
- **Equity Curve**: Visualize account balance and equity progression over time
- **Raw Data Access**: Examine the underlying trade data with flexible filtering options

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up MySQL database credentials in `.streamlit/secrets.toml`
4. Run the application: `streamlit run main.py`

## Usage

1. Launch the dashboard
2. Upload your cTrader CSV backtest export file
3. Adjust risk parameters in the sidebar as needed
4. Navigate through the tabs to explore different aspects of your trading performance

## Requirements

- Python 3.8+
- MySQL database
