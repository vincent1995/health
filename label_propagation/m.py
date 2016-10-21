try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import label_propagation

#######################################################
# 载入数据

# 载入样本标签
f = open(r"..\feature_extract\landmark_data",'rb')
data = pickle.load(f)
all_labels = data[2]
flags = data[0]

# 载入肤色特征
f = open(r"..\feature_extract\feature_data", 'rb')
face_color_feature = pickle.load(f)
f.close()

# 载入lbp特征
f = open(r"..\feature_extract\lbp_feature", 'rb')
lbp_feature = pickle.load(f)
f.close()

# 组合特征
features = np.hstack((face_color_feature,lbp_feature))
# features = np.array(face_color_feature)
# z-score 特征归一化
mean = np.mean(features,0)
var = np.var(features,0)
features = (features -mean)/var
features = features.tolist()
# print(features[100])

labeledNum = int(len(features)*0.05)
Mat_labeled = np.array(features[:labeledNum])
Mat_unlabeled = np.array(features[labeledNum:])

for type in range(1,10):
    labels = []
    for label in all_labels:
        if str(type) in label:
            labels.append(1)
        else:
            labels.append(0)
    unlabel_data_labels = label_propagation.labelPropagation(Mat_labeled, Mat_unlabeled, labels[:labeledNum], kernel_type = 'knn', knn_num_neighbors = 10, max_iter = 400)
    TP = 0
    for i, label in enumerate(labels[labeledNum:]):
        if label == unlabel_data_labels[i] and label == 1:
            TP+=1
    if sum(unlabel_data_labels) == 0:
        pre = float('nan')
    else:
        pre = TP / sum(unlabel_data_labels)
    if sum(labels[labeledNum:]) == 0:
        rec = float('nan')
    else:
        rec = TP/sum(labels[labeledNum:])
    print(pre,rec, 2*pre*rec/(pre+rec))
