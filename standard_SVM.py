# 标准的SVM方法，不同的特征都使用该SVM实验，从而客观的评价各种方法。
# 10-fold 交叉验证，测试集验证集均10-fold
# SVM参数选择：

from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 参数调整范围
turned_parameter = [{'C': [1, 10, 100, 1000]}]


# 运行SVM
# 输入参数：特征list和标签list
def run_SVM(X, y):
    cm, acc = 0, 0
    # 特征归一化
    X = scale(X)
    # 10-fold 测试集
    kf = KFold(n_splits=10,shuffle=True,random_state=43)
    for not_test_index, test_index in kf.split(X):
        X_not_test, X_test = X[not_test_index], X[test_index]
        y_not_test, y_test = y[not_test_index], y[test_index]
        # 10-fold 验证集, 获得最优参数
        clf = GridSearchCV(SVC(), turned_parameter, cv=10)
        clf.fit(X_not_test, y_not_test)
        # 在测试集运行
        y_predict = clf.predict(X_test)
        cm += confusion_matrix(y_test, y_predict)
        acc += accuracy_score(y_test, y_predict)
    cm = cm/10
    cm = cm/np.sum(cm,axis=1).reshape((cm.shape[0],1))
    acc /= 10
    return cm, acc

# 标准绘制混淆矩阵函数
# 输入：混淆矩阵，标签名list
def plot_confusion_matrix(cm, label_names=None, title='Confusion matrix',cmap=plt.cm.gray):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_names != None:
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X,y =make_classification(400,20,n_classes=8,n_informative=6)
    cm,acc = run_SVM(X,y)
    plot_confusion_matrix(cm)

