import iexfinance as ie
from datetime import datetime
from iexfinance.stocks import get_historical_data
from iexfinance import get_market_book
from iexfinance import StockReader
from iexfinance import get_available_symbols

#symbols = get_available_symbols(output_format='pandas')
#print(len(symbols))
start = datetime(2017, 1, 1)
end = datetime(2018, 1, 1)
df = get_historical_data("AA", start, end)
print(df)