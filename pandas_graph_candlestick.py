import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline

offline.init_notebook_mode(connected=True)

stock_data = pd.read_csv('datasets/stock_market.csv')

trace = go.Candlestick(x = stock_data.Date,

                       open = stock_data.Open,

                       high = stock_data.High,

                       low = stock_data.Low,

                       close = stock_data.Close)

data = [trace]

offline.plot(data)

trace = go.Candlestick(x = stock_data.Date[0:10],

                       open = stock_data.Open[0:10],

                       high = stock_data.High[0:10],

                       low = stock_data.Low[0:10],

                       close = stock_data.Close[0:10])

data = [trace]

offline.plot(data)
