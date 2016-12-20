from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def param_select(X, y):
    # 参数范围
    params = {'c':[]}
    # 特征归一化
    X = scale(X)
    # 运行SVM
    param = 'c'
    scores = []
    for p in params[param]:
        acc = cross_val_score(SVC(C=p),X,y,cv=10)
        scores.append(acc)
    # 绘制图表
