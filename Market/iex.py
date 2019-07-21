import iexfinance as ie
from datetime import datetime
from iexfinance.stocks import get_historical_data
from iexfinance import get_market_book
#from iexfinance import StockReader
from iexfinance.refdata import get_symbols
from iexfinance.stocks import Stock
import os

os.environ['IEX_TOKEN'] = 'sk_91ee4fb3cf4e4eecb06b710941644e62'
symbols = get_symbols(output_format='pandas')
symbols.to_csv('symbols.csv')
start = datetime(2014, 1, 1)
end = datetime(2015, 1, 1)
#df = get_historical_data("AA", start, end)
#print(df)