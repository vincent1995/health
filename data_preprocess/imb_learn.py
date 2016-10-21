from imblearn.over_sampling import SMOTE
import numpy as np

def smote(features,labels):
    # 获得最大数量标签及其样本数
    bincount = np.bincount(labels)
    # print(bincount)
    max_label = np.argmax(bincount)
    max_label_num = bincount[max_label]
    # 初始化数据
    x_reshape = []
    sample_size = labels.shape[0]
    sm = SMOTE()
    # 依次生成每个种类的样本
    for type in range(1,10):
        # 如果该种类为标签数最多的种类，跳过
        if type == max_label:
            x_reshape.append(None)
            continue
        else:
            print(type)
            y = (labels == type)
            y.dtype = np.uint8
            x_r,y_r = sm.fit_sample(features,y)
            print(sample_size,x_r.shape)
            x_reshape.append(x_r[sample_size:])
    # 将新样本添加进样本集中，保证每个类别样本数量都相等
    labels = labels.tolist()
    for type in range(1,10):
        if type == max_label:
            continue
        # 计算需要添加的样本数量
        num = max_label_num - np.sum(labels == type)
        np.random.shuffle(x_reshape[type-1])
        X = x_reshape[type-1][:num]
        # features = np.array(features)
        # X = np.array(X)
        features = np.row_stack((features,X))
        labels+= (np.ones(num)*type).tolist()
    labels = np.array(labels)
    print('finish')
    return features,labels

