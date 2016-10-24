if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import SVC
    import itertools
    from openpyxl import Workbook
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    from imblearn.over_sampling import SMOTE
    # 载入所有特征，每个特征是一个dict
    features_name = [
        # "face_main_color",\
        'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
        # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power', \
         'small_tongue_color_hist', 'small_tongue_color_rect', 'small_tongue_gray_texture', 'small_tongue_lbp', 'small_tongue_power', \
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

    # 生成所有特征的所有可能组合
    used_features = []
    for i in range(1,len(features_set)+1):
        used_features += list(itertools.combinations(features_set.keys(),i))
    # 使用一种特征组合
    used_features = [ [\
                    'face_color_hist','face_color_rect','face_gray_texture','face_power','face_lbp',\
                    'small_tongue_color_hist', 'small_tongue_color_rect', 'small_tongue_gray_texture', 'small_tongue_lbp','small_tongue_power', \
                    # 'tongue_color_hist','tongue_color_rect','tongue_gray_texture','tongue_lbp','tongue_power',\
                     'face_block_color_hist','face_block_color_rect','face_block_gray_texture','face_block_lbp','face_block_power']]
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
        # 切分数据集，交叉验证多次计算
        measures = np.zeros(1)
        cal_num = 5 # 数据切分的份数
        run_num = 5 # svm运行次数
        sample_num = features.shape[0]
        # 标签分布
        bitcount = np.zeros(9)
        for cal in range(0,cal_num):
            if cal == run_num:
                break;
            l_edge = int(sample_num*cal/cal_num)
            r_edge = int(sample_num*(cal+1)/cal_num)
            test_features = features[l_edge:r_edge,:]
            test_labels = labels[l_edge:r_edge]
            train_features = np.row_stack((features[0:l_edge,:],features[r_edge:sample_num,:]))
            train_labels = np.array(labels[0:l_edge].tolist()+labels[r_edge:sample_num].tolist())
            # 使用 smote
            # smote = SMOTE(random_state=42)
            # train_features,train_labels = smote.fit_sample(train_features,train_labels)
            # 设定svm参数
            cw = {}
            max_labels_num = np.max(np.bincount(train_labels))
            for t in range(1,10):
                cw[t] = max_labels_num / np.bincount(train_labels)[t]
            svm = SVC(class_weight=cw,C=1000)
            # svm训练
            # module = svmutil.svm_train(train_labels.tolist(),
            #                            train_features.tolist(),
            #                            '-q')
            svm.fit(train_features,train_labels)
            # svm验证
            # predict_label,_,_ = svmutil.svm_predict(test_labels.tolist(),
            #                                  test_features.tolist(),
            #                                  module,
            #                                  '-q')
            predict_label = svm.predict(test_features)
            # 统计在该测试集上的准确率
            acc = np.sum(test_labels == predict_label) / len(predict_label)
            measures += acc
            # 混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(test_labels,predict_label)
            print(cm)
            # 统计label分布
            # print("predicted",np.bincount(predict_label,minlength=10)[1:])
            # print("answer",np.bincount(test_labels, minlength=10)[1:])
            # print("correct",np.bincount((test_labels == predict_label)*1*test_labels, minlength=10)[1:])
            # print(predict_label)
            # print('总样本数：', len(labels))
            # labels_distribution = []
            # for i in range(1,10):
            #     num = np.sum(np.array(labels) == i)
            #     labels_distribution.append(num)
            # print('样本标签分布：',labels_distribution)
            # print('测试样本数：', len(predict_label))
            # labels_distribution = []
            # for i in range(1, 10):
            #     num = np.sum(np.array(predict_label) == i)
            #     labels_distribution.append(num)
            # print('样本标签分布：', labels_distribution)
            # 输出进度提示
            print('order all:', len(used_features), 'cur:', cur_num, 'accuracy:', acc)
        # 求不同数据集测试结果平均值
        measures /= run_num
        # 获得所使用特征名称
        features_name = ''
        for feature in used_feature:
            features_name+=feature+','
        # 计算标签分布
        # 保存结果到xmls文件
        wb.active.append([features_name]+[str(measures.tolist())])
        print(measures.tolist())
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


