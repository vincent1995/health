if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import SVC
    import itertools
    from openpyxl import Workbook
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    from sklearn.metrics import confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    # 载入所有特征，每个特征是一个dict
    features_name = [
        # "face_main_color",\
        'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
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

    # 生成所有特征的所有可能组合
    used_features = []
    for i in range(1,len(features_set)+1):
        used_features += list(itertools.combinations(features_set.keys(),i))
    # 使用一种特征组合
    used_features = [ [\
                    'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
                    # 'small_tongue_color_hist', 'small_tongue_color_rect', 'small_tongue_gray_texture', 'small_tongue_lbp','small_tongue_power', \
                    # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power',\
                    #  'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power'
                    ]]
    # used_features = [['face_main_color','tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','face_block_color_rect','face_block_gray_texture','face_block_lbp']]
    # 生成结果存储文件
    wb = Workbook()
    wb.active.append(['所用特征','accuracy','标签1在预测结果中所占比例'])

    # 对每种特征组合进行svm运算
    for cur_num,used_feature in enumerate(used_features):
        # 获得特征子集和label子集
        features = []
        labels = []
        for patient in patient_list:
            ID = patient[0]
            feature = []
            flag = True
            for i in used_feature:
                if ID in features_set[i].keys():
                    new_feature = features_set[i][ID]
                    feature+=new_feature
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
        features = (features - np.mean(features)) / (np.std(features,axis=0)+1e-9)
        ####################################
        ## n-fold SVM
        # from sklearn.model_selection import cross_val_score, cross_val_predict
        # num_fold = 5
        # svm = SVC(class_weight="balanced",C=1000)
        # score = cross_val_score(svm,features,labels,cv=num_fold)
        # print(score)
        ####################################
        # 1-fold
        from sklearn.model_selection import train_test_split
        train_x,test_x,train_y,test_y = train_test_split(features,labels,test_size=466,random_state=42)
        svm = SVC(class_weight="balanced",C=1000)
        svm.fit(train_x,train_y)
        predict_y = svm.predict(test_x)

        # lg = LogisticRegression(solver="newton-cg",multi_class="ovr",max_iter=1000)
        # lg.fit(train_x,train_y)
        # predict_y = lg.predict(test_x)
        # 混淆矩阵
        cm = confusion_matrix(test_y,predict_y)
        print(cm)
        acc = np.sum(predict_y==test_y)/predict_y.shape[0]
        print(acc)
        #######################################
        # 获得所使用特征名称
        features_name = ''
        for feature in used_feature:
            features_name+=feature+','
        # 计算标签分布
    try:
        wb.save(path+r"\single_label_运行结果2.xlsx")
        pass
    except PermissionError:
        print("关闭excel文件继续")
        import dlib
        dlib.hit_enter_to_continue()
        wb.save(path + r"\single_label_运行结果.xlsx")

def test():
    print("test")


