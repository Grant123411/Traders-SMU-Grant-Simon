import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


stocks = ["AAPL", "DELL", "META", "NVDA", "GOOGL", "WMT", "MCD", "KO", "SWK", "ULS"]


def calculate_implied_volatility(ticker):
    stock_data = yf.download(ticker, start="2024-01-01", end="2025-03-05", auto_adjust=False, multi_level_index=False)
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    daily_volatility = stock_data['Daily Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility
volatility_dict = {stock: calculate_implied_volatility(stock) for stock in stocks}
sorted_volatility = sorted(volatility_dict.items(), key=lambda x: x[1], reverse=True)
print("Stocks sorted by implied volatility (descending):")
for stock, volatility in sorted_volatility:
    print(f"{stock}: {volatility:.4f}")
top_5_stocks = sorted_volatility[:5]
print("\nTop 5 most volatile stocks for further analysis:")
for stock, volatility in top_5_stocks:
    print(f"{stock}: {volatility:.4f}")



import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = ["AAPL", "DELL", "META", "NVDA", "GOOGL", "WMT", "MCD", "KO", "SWK", "ULS"]

class BollingerBandsTrader:
    def __init__(self, data, ticker):
        self.data = data.copy()
        self.ticker = ticker
        self._flatten_columns()  
        self.calculate_bollinger_bands()

    def _flatten_columns(self):
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(1)

    def calculate_bollinger_bands(self):
        self.data['SMA'] = self.data['Adj Close'].rolling(window=20).mean()
        self.data['Rolling Std'] = self.data['Adj Close'].rolling(window=20).std()
        self.data['Upper Band'] = self.data['SMA'] + (2 * self.data['Rolling Std'])
        self.data['Lower Band'] = self.data['SMA'] - (2 * self.data['Rolling Std'])

    def generate_signals(self):
        self.data['Signal'] = np.where(self.data['Adj Close'] > self.data['Lower Band'], 'Buy', 'Hold')
        self.data['Signal'] = np.where(self.data['Adj Close'] < self.data['Upper Band'], 'Sell', self.data['Signal'])

    def backtest_strategy(self):
        # Calculate daily returns based on adjusted close
        self.data['Returns'] = self.data['Adj Close'].pct_change()
        self.data['Strategy Returns'] = np.where(self.data['Signal'] == 'Buy', self.data['Returns'], 0)
        self.data['Cumulative Strategy Returns'] = (1 + self.data['Strategy Returns']).cumprod()
        final_return = self.data['Cumulative Strategy Returns'].iloc[-1]
        
        
        daily_mean = self.data['Strategy Returns'].mean()
        daily_std = self.data['Strategy Returns'].std()
        standardized_return = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else np.nan
        
        
        percentage_return = (final_return - 1) * 100
        
        
        print(f"\n{self.ticker} Strategy Return (Cumulative): {final_return:.2f}")
        print(f"{self.ticker} Standardized Return (Annualized Sharpe Ratio, assuming 0 risk-free rate): {standardized_return:.2f}")
        print("Standardized returns measure the risk-adjusted performance by normalizing the average daily return by its volatility (annualized).")
        print(f"If you invested $1, your final amount would be ${final_return:.2f}, which is a {percentage_return:.2f}% return.\n")
        
        return final_return

    def visualize_strategy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Adj Close'], label='Price', color='blue')
        plt.plot(self.data['Upper Band'], label='Upper Band', color='red', linestyle='--')
        plt.plot(self.data['Lower Band'], label='Lower Band', color='green', linestyle='--')
        plt.scatter(self.data.index, self.data['Adj Close'],
                    c=np.where(self.data['Signal'] == 'Buy', 'green', 'red'),
                    label='Signals', marker='^', alpha=0.7)
        plt.title(f'{self.ticker} Bollinger Bands & Buy/Sell Signals')
        plt.legend()
        plt.show()


final_returns = []


for stock in stocks:
    
    stock_data = yf.download(stock, start="2024-01-01", end="2025-03-05", auto_adjust=False, progress=False, group_by='ticker')
    
    
    trader = BollingerBandsTrader(stock_data, stock)
    
    
    trader.generate_signals()
    
   
    final_return = trader.backtest_strategy()
    final_returns.append(final_return)
    
    
    trader.visualize_strategy()


percentage_returns = [(ret - 1) * 100 for ret in final_returns]  
average_percent_return = np.mean(percentage_returns)
total_final_amount = sum(final_returns)  
total_money_made = total_final_amount - len(stocks)  

print("\nSummary of All Stocks:")
print(f"Average Percent Return: {average_percent_return:.2f}%")
print(f"Total Final Amount (if $1 invested per stock): ${total_final_amount:.2f}")
print(f"Total Money Made (Profit): ${total_money_made:.2f}")


#The volatility of the stocks was pretty interesting, especially when looking at how much their prices fluctuated. Stocks like DELL, 
# NVDA, META, SWK, and GOOGL were the most volatile, with DELL leading the pack at 0.5910. These stocks saw bigger price swings, meaning
# they came with higher risk, but also the chance for bigger rewards. Volatility definitely played a big part in this analysis, as it 
# helps us understand the kind of risks we’re taking and whether a stock is worth adding to a portfolio.

#When it comes to the Bollinger Bands strategy, the performance seemed pretty good overall. DELL, for example, gave the highest return 
# at 2.36x, which shows that the strategy worked well for stocks with higher volatility. NVDA and META also did well, but stocks like 
# MCD and KO didn’t perform as strongly, which probably means the strategy works better on stocks with more price movement. So, the 
# strategy really seemed to shine when there were larger swings in price, but it wasn’t always the best choice for more stable stocks.

#Looking at the graphs, it was clear that the strategy worked well when prices broke out of the Bollinger Bands, which usually signaled 
# a bigger price movement. That said, there were some moments where just holding onto the stocks without making any moves could have 
# worked better, especially for those less volatile ones. But overall, the analysis showed that volatility is key when it comes to 
# making the most out of strategies like Bollinger Bands. If a stock moves a lot, that’s where you’ll see the best results from the 
# strategy.
