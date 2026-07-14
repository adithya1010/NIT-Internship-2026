# 🚀 Deep Portfolio: Multi-Modal Transformer & LSTM Asset Allocation

![Python](https://img.shields.io/badge/python-3.10+-blue.svg) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyPortfolioOpt](https://img.shields.io/badge/Library-PyPortfolioOpt-green.svg)

This project implements an advanced quantitative finance pipeline that bridges **Deep Learning** with **Modern Portfolio Theory (MPT)**. It evolves from a standard LSTM forecasting model to a state-of-the-art **Multi-Modal Transformer** architecture to optimize a 99-asset universe.

## 📌 Project Overview
Traditional portfolio optimization (Mean-Variance Optimization) often fails due to the 'Linearity Assumption'—the idea that historical returns perfectly predict future ones. This project replaces static averages with **predictive neural intelligence**.

### Key Features
- **Hybrid Architecture:** Combines Self-Attention Transformer encoders with Efficient Frontier optimization.
- **Multi-Modal Inputs:** Processes both historical price streams and alternative macroeconomic/sentiment indices (e.g., VIX Proxy).
- **Friction-Aware Optimization:** Includes L1-norm penalties for transaction costs (turnover) to ensure realistic backtesting.
- **Walk-Forward Validation:** Implements a strict rolling window validation engine to prevent data leakage.
- **Custom Financial Loss:** Uses a differentiable Sharpe/Sortino proxy loss function to optimize for risk-adjusted returns during backpropagation.

## 🏗️ Technical Architecture
1.  **Data Layer:** Downloads live ticker data via `yfinance` and generates synthetic institutional-grade datasets (99 assets, 200+ days).
2.  **Forecasting Engine (Transformer):** Utilizes Multi-Head Attention to capture long-range temporal dependencies and cross-asset correlations.
3.  **Optimization Layer:** Injects Transformer price forecasts into a customized `EfficientFrontier` engine with EWMA risk models and downside semivariance constraints.

## 📊 Performance Summary
Based on the institutional backtest results:
- **Strategic Wealth Factor:** ~1.26 (26% out-of-sample growth).
- **Risk Management:** Successfully navigated synthetic volatility clusters with a maximum drawdown of only **-9.06%**.
- **Forecasting Accuracy:** Achieved a Mean Absolute Error (MAE) of ~0.19 across a high-dimensional asset cloud.

## 🛠️ Installation & Usage
```bash
pip install numpy pandas yfinance tensorflow pyportfolioopt scikit-learn matplotlib
To run the pipeline, execute the main validation loop in the provided notebook. The system is modular, allowing you to swap all_ten.csv or synthetic generators for live Bloomberg/Reuters data feeds.

📜 License
Distributed under the MIT License. See LICENSE for more information. ```
