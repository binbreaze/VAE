import pandas as pd

x = pd.read_csv('D:\\taocloud\donut-master\sample_data\cpu4.csv')

# value = x['value']
# label = x['label']
# print(label[:10])
num = x.ix[(x['label'] > 0),['label']].count()
print(num)