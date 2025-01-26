![Logo](https://s3.timeweb.cloud/68597a50-pictrace/photo_2024-11-12_03-23-43.jpg)

-----------------

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/README.md">⭐English⭐</a> |
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/CryptoBoat/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

## ⚠️ IMPORTANT DISCLAIMER ⚠️
### This trading bot is currently in EXPERIMENTAL/BETA testing phase. By using this software:
1. **Cryptocurrency Trading Risks:** You acknowledge that trading cryptocurrencies involves substantial risks, including the potential loss of your invested capital.
2. **Technological Limitations:** The bot utilizes Artificial Intelligence and Machine Learning (AI/ML) models that are still undergoing testing and improvements. This may lead to unforeseen errors or inaccurate signals.
3. **Liability for Losses:** You accept full responsibility for any financial losses that may occur as a result of using this bot.
4. **No Guarantee of Performance:** Past performance does not guarantee future results. The cryptocurrency market is highly volatile and can change rapidly.
5. **Capital Management:** Trade only with funds you can afford to lose. Do not invest money that is necessary for your living expenses or other essential purposes.
6. **No Financial Advice:** This software is NOT financial advice. Use it at your own risk and consult with professional financial advisors before making investment decisions.

## Support the Project 💖

If you've found this project helpful or profitable, consider supporting its development:

<div align="center">
  <img src="https://github.com/Solrikk/Small-Boats/blob/main/assets/photo/qrcode.png" alt="Donation QR Code" width="200"/>
</div>

Your donations help maintain and improve the project, ensuring its continued development and enhancement.

# Small-Boats 🚀

----

## Overview 📊

**CryptoBoat** is a **semi-automated cryptocurrency trading system** that emphasizes manual control over key trading decisions while managing multiple positions simultaneously. Unlike fully automated bots, this strategy allows you to:

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FSolrikk%2FSmall-Boats&label=Views&countColor=%232ccce4)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FSolrikk%2FSmall-Boats)
[![GitHub stars](https://img.shields.io/github/stars/Solrikk/Small-Boats?style=flat&logo=github&color=yellow)](https://github.com/Solrikk/Small-Boats/stargazers)
[![Profile Views](https://komarev.com/ghpvc/?username=Solrikk&color=brightgreen&style=flat&label=Profile+Views)](https://github.com/Solrikk)
[![GitHub license](https://img.shields.io/github/license/Solrikk/Small-Boats?color=blue&style=flat)](https://github.com/Solrikk/Small-Boats/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat&logo=python)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-modern-green?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)


- Manually control Take-Profit and Stop-Loss levels
- Open multiple strategic positions across different assets
- Benefit from risk diversification through position spreading
- Offset potential losses with gains from other positions

## Technical Details 🔧

### AI Models Used
1. **LSTM (Long Short-Term Memory)**
   - Specialized neural network for time series prediction
   - Capable of learning long-term dependencies
   - Optimized for cryptocurrency price movement patterns
   - Uses multiple technical indicators for enhanced accuracy

2. **Random Forest**
   - Ensemble learning algorithm
   - Combines multiple decision trees
   - Reduces overfitting through aggregation
   - Provides robust market trend predictions

### Technical Indicators
- RSI (Relative Strength Index)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Ichimoku Cloud
- VWAP (Volume Weighted Average Price)
- ATR (Average True Range)

## Performance Features 🎯

- Real-time market data processing
- Multi-timeframe analysis
- Advanced risk management system
- Position size optimization
- Automated entry/exit signals
- Portfolio rebalancing
- Custom indicator combinations

## Technical Architecture 🔧

### System Components
1. **Neural Networks**
   - LSTM Network: 2-layer bidirectional architecture
   - Input shape: (60, 18) - 60 timeframes, 18 features
   - Hidden layers: 100 units each with dropout (0.3)
   - Output: Binary classification (buy/sell signal)

2. **Random Forest Classifier**
   - Ensemble of 100 decision trees
   - Feature flattening: 1080 dimensions (60 timeframes × 18 features)
   - Class balancing with SMOTE
   - Parallel prediction processing

3. **Technical Indicators**
   - Price-based: EMA, Bollinger Bands, Ichimoku
   - Momentum: RSI, MACD, Stochastic
   - Volume: VWAP
   - Volatility: ATR

## Installation & Setup 🛠️

### Prerequisites
- Python 3.8+
- GPU recommended for faster model training
- Minimum 4GB RAM
- Bybit account with API access

### Step-by-Step Installation

1. **Initialize Project:**
```bash
git clone https://github.com/Solrikk/CryptoBoat.git
cd CryptoBoat
```

2. **Install Required Packages:**
```bash
pip install numpy pandas tensorflow scikit-learn ta ccxt matplotlib
```

3. **Configure Exchange:**
Update `main.py` with your Bybit API credentials:
```python
API_KEY = "your_api_key"    # From Bybit dashboard
API_SECRET = "your_api_secret"
```

4. **Configure Risk Parameters:**
In `main.py`, adjust trading parameters:
```python
risk_percentage = 0.3  # Risk per trade (0.3%)
TRADE_COOLDOWN = 60   # Seconds between trades
```

### Running the Bot

1. **Initial Launch:**
```bash
python main.py
```

2. **Monitor Operations:**
- Check `trading_bot_derivatives.log` for real-time status
- Review `trades_log.json` for trade history

### System Architecture

```
Input Data → Feature Engineering → Model Prediction → Trading Logic
    ↓              ↓                    ↓               ↓
OHLCV Data → Technical Indicators → LSTM + RF → Position Management
```

### Performance Monitoring
- Log files track all operations
- Real-time balance updates
- Trade execution confirmations
- Error handling and reporting

### Risk Management
- Position sizing based on account balance
- Maximum risk per trade: 0.3%
- Trade cooldown period: 60 seconds
- Automatic error recovery
- Multiple validation layers

### Error Handling
- Network disconnect protection
- API error recovery
- Invalid data detection
- Balance verification
- Order validation

## Usage Guide 📚

1. Start the Bot:
```bash
python main.py
```

2. Monitor the Logs:
- Check `trading_bot_derivatives.log` for detailed operation logs
- Review trade history in `trades_log.json`

3. Configure Risk Parameters:
- Adjust position sizes in `calculate_position_size()`
- Modify risk percentage (default: 0.3%)
- Set custom stop-loss levels

⚠️ **Remember**: Past performance does not guarantee future results. Always trade responsibly and within your risk tolerance.

## Community & Support 🤝

- Join our community discussions
- Share your trading strategies
- Report issues and suggest improvements
- Help others get started

