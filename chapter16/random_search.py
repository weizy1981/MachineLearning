from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': uniform()}
# 通过网格搜索查询最优参数
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
grid.fit(X, Y)
# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s' % grid.best_estimator_.alpha)