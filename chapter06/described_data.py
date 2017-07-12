from pandas import read_csv
from pandas import set_option
# 描述性统计
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
# 设置数据的精确度
set_option('precision', 4)
print(data.describe())