import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import standard_SVM



if __name__ == '__main__':
    # 载入特征，每个特征是一个dict
    features_name = [
        'face_block_avg_color'
        # "face_main_color",\
        # 'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
        # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power', \
        #  'small_tongue_color_hist', 'small_tongue_color_rect', 'small_tongue_gray_texture', 'small_tongue_lbp', 'small_tongue_power', \
        #  'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power'
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
    # 组合特征
    features = []
    labels = []
    for patient in patient_list:
        ID = patient[0]
        label = patient[1]
        feature = []
        flag = True
        for i in features_name:
            if ID in features_set[i].keys():
                new_feature = features_set[i][ID]
                feature+=list(new_feature)
            else:
                flag = False
                break
        if flag:
            labels.append(patient[1])
            features.append(feature)
    features = np.array(features)
    labels = np.array(labels,dtype=np.int8)
    print(len(features))
    print(np.bincount(labels))
    # 运行SVM
    cm,acc =standard_SVM.run_SVM(features,labels)
    print(cm)
    # 绘制混淆矩阵
    standard_SVM.plot_confusion_matrix(cm,['体质1','体质2','体质3','体质4',\
                                           '体质5','体质6','体质7','体质8'])
    print("accuracy:",acc)



