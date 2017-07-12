from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
cart = DecisionTreeClassifier()
models = []
model_logistic = LogisticRegression()
models.append(('logistic', model_logistic))
model_cart = DecisionTreeClassifier()
models.append(('cart', model_cart))
model_svc = SVC()
models.append(('svm', model_svc))
ensemble_model = VotingClassifier(estimators=models)
result = cross_val_score(ensemble_model, X, Y, cv=kfold)
print(result.mean())
