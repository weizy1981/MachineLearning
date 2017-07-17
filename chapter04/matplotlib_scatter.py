import matplotlib.pyplot as plt
import numpy as np

# 定义绘图的数据
myarray1 = np.array([1, 2, 3])
myarray2 = np.array([11, 21, 31])
# 初始化绘图
plt.scatter(myarray1, myarray2)
# 设定x，y轴
plt.xlabel('x axis')
plt.ylabel('y axis')
# 绘图
plt.show()