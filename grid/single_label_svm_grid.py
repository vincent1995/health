import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# 载入所有特征，每个特征是一个dict
features_name = [\
                # 'face_block_avg_color'
                 'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
                 # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power',\
                 # 'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power'
                # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','face_block_color_rect','face_block_gray_texture','face_block_lbp',
                ]
features_set = {}
path = r'B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\4466'
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
            feature += new_feature
        else:
            flag = False
            break
    if flag:
        labels.append(patient[1])
        features.append(feature)
features = np.array(features)
labels = np.array(labels,dtype=np.int8)
print(np.bincount(labels))
print(len(features))
# 特征归一化
from sklearn.preprocessing import scale
features = scale(features)
# 切分数据集
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=466,random_state=42)
measures = np.zeros(1)
# 设定svm参数
# cw = {}
# max_labels_num = np.max(np.bincount(train_labels))
# for type in range(1, 10):
#     cw[type] = max_labels_num / np.bincount(train_labels)[type]
# grid_search
turned_parameter = {

    }
# # from sklearn.linear_model import LogisticRegression
# clf = GridSearchCV(SVC(C=1000),turned_parameter,cv=5)
# clf.fit(train_features,train_labels)
# print(clf.best_params_,clf.best_score_)
# # svm验证
# predict_label = clf.predict(test_features)
# print(np.bincount(predict_label))

svm = SVC(C=1000)
svm.fit(train_features,train_labels)
predict_label = svm.predict(test_features)
# 混淆矩阵
cm = confusion_matrix(test_labels, predict_label)
print(cm)
print("acc",metrics.accuracy_score(test_labels,predict_label))
# print("recall",metrics.recall_score(test_labels,predict_label))
# print("precision",metrics.precision_score(test_labels,predict_label))
# print("f1",metrics.f1_score(test_labels,predict_label))