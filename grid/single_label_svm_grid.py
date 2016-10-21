import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
try:
    import cPickle as pickle
except ImportError:
    import pickle


# 载入所有特征，每个特征是一个dict
features_name = [\
                 'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
                 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power',\
                 'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power']
features_set = {}
path = r'B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin'
for feature_name in features_name:
    f = open(path+'\\'+feature_name,'rb')
    data = pickle.load(f)
    features_set[feature_name]=data
# 载入所有患者ID 和 标签
f = open(path + r"\labels",'rb')
data = pickle.load(f)
# 改变label表示方式
patient_list=[]
for d in data:
    p = []
    p.append(d[0])
    p.append(d[1])
    patient_list.append(p)
patient_list = np.array(patient_list)
np.random.shuffle(patient_list)
# 获得特征子集和label子集
features = []
labels = []
for patient in patient_list:
    ID = patient[0]
    feature = []
    flag = True
    for i in features_name:
        if ID in features_set[i].keys():
            new_feature = features_set[i][ID]
            if (type(new_feature) != list):
                new_feature = new_feature.tolist()
            feature += new_feature
        else:
            flag = False
            break
    if flag:
        labels.append(patient[1])
        features.append(feature)
features = np.array(features)
labels = np.array(labels,dtype=np.int8)
print(len(features))
# 特征归一化
features = (features - np.mean(features)) / (np.std(features, axis=0) + 1)
# 切分数据集，交叉验证多次计算
train_features,test_features,train_labels,test_labels = train_test_split(features,labels)
measures = np.zeros(1)
# 设定svm参数
# cw = {}
# max_labels_num = np.max(np.bincount(train_labels))
# for type in range(1, 10):
#     cw[type] = max_labels_num / np.bincount(train_labels)[type]
# grid_search
turned_parameter = {'alpha': 10.0 ** -np.arange(1, 7),'solver': ['lbfgs', 'sgd', 'adam'],\
                    'hidden_layer_sizes':[(100),(100,100),(100,100,100),(100,100,100,100)]}
from sklearn.neural_network import MLPClassifier
clf = GridSearchCV(MLPClassifier(),turned_parameter,cv=5)
clf.fit(train_features,train_labels)
print(clf.best_params_,clf.best_score_)
# svm验证
predict_label = clf.predict(test_features)
print(np.bincount(predict_label))

# for i in [1000]:
#     svm = SVC(class_weight="balanced",C=i)
#     svm.fit(train_features,train_labels)
#     predict_label = svm.predict(test_features)
#     # 统计在该测试集上的准确率
#     acc = np.sum(test_labels == predict_label) / len(predict_label)
#     print(i,acc)
#     print(np.bincount(predict_label))
#     print(np.bincount(test_labels))
