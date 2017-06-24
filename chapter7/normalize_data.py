# 标准化数据
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = Normalizer().fit(X)
# 数据转换
rescaledX = scaler.transform(X)
# 设定数据的打印格式
set_printoptions(precision=3)
print(rescaledX)
