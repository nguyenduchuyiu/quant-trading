# plot close price of data 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/full_data.csv')

# convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# set date as index
df.set_index('date', inplace=True)

# plot close price
plt.plot(df['close'], label='Close Price')
plt.title('Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()