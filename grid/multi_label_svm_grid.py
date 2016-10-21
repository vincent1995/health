import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
try:
    import cPickle as pickle
except ImportError:
    import pickle


# 载入所有特征，每个特征是一个dict
features_name = ['face_main_color','tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power',\
                 'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power']
features_set = []
for feature_name in features_name:
    f = open(r'C:\offline\体质辨识数据备份\origin\\'+feature_name,'rb')
    data = pickle.load(f)
    dict = {}
    for d in data:
        if not d==None:
            dict[int(d[0])] = d[1:]
    features_set.append([feature_name,dict])
# 载入所有患者ID 和 标签
f = open(r"C:\offline\体质辨识数据备份\origin\labels",'rb')
data = pickle.load(f)
# 改变label表示方式
patient_list=[]
for d in data:
    p = []
    p.append(int(d[0]))
    for i in range(1,10):
        p.append(str(i) in d[1:])
    patient_list.append(p)
patient_list = np.array(patient_list)

# 进行svm运算

# 获得特征子集和label子集
features = []
labels = []
for patient in patient_list:
    ID = patient[0]
    feature = []
    flag = True
    for i in range(len(features_set)):
        if ID in features_set[i][1].keys():
            feature+=features_set[i][1][ID]
        else:
            flag = False
            break
    if flag:
        labels.append(patient[1:].tolist())
        features.append(feature)
features = np.array(features)
labels = np.array(labels)
# 切分数据集，交叉验证多次计算
train_features,test_features,train_labels,test_labels = train_test_split(features,labels)
predict_label = []
# 9中体质类型分别运算
for type in range(0,9):
    # svm训练
    svm = SVC(class_weight='balanced')
    svm.fit(train_features,train_labels[:,type])
    # svm验证
    pre_label = svm.predict(test_features)
    score = svm.score(test_features,test_labels[:,type])
    predict_label.append(pre_label.tolist())
    print(type,score,np.bincount(pre_label))
predict_label = np.transpose(np.array(predict_label))
# 统计在该测试集上的准确率
# 参考 http://blog.csdn.net/bemachine/article/details/10471383 上的多标签分类评价标准
acc = np.sum( np.sum(np.all([predict_label,test_labels],axis=0),axis=1) / np.sum(np.any([predict_label,test_labels],axis=0),axis=1) )/predict_label.shape[0]
pre = np.sum(np.sum(np.all([predict_label,test_labels],axis=0), axis=1) / np.sum(predict_label, axis=1)) /predict_label.shape[0]
rec = np.sum(np.sum(np.all([predict_label,test_labels],axis=0), axis=1) / np.sum(test_labels, axis=1)) /predict_label.shape[0]
f = 2*pre*rec/(pre+rec)
print(acc,pre,rec,f)


