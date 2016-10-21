import numpy as np
from svmutil import svmutil
import itertools
from openpyxl import Workbook
import subprocess
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

# 生成所有特征的所有可能组合
orders = []
for i in range(1,len(features_set)+1):
    orders += list(itertools.combinations(range(len(features_set)),i))

# 生成结果存储文件
wb = Workbook()
wb.active.append(['所用特征','accuracy','precision','recall','F值'])

# 对每种特征组合进行svm运算
for order_num,order in enumerate(orders):
    # 获得所使用特征名称
    features_name = ''
    for i in order:
        features_name+=(features_set[i][0]+',')
    # 获得特征子集和label子集
    features = []
    labels = []
    for patient in patient_list:
        ID = patient[0]
        feature = []
        flag = True
        for i in order:
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
    measures = np.zeros(4)
    cal_num = 5 # 数据切分的份数
    run_num = 1 # svm运行次数
    sample_num = features.shape[0]
    for cal in range(0,cal_num):
        if cal == run_num:
            break;
        l_edge = int(sample_num*cal/cal_num)
        r_edge = int(sample_num*(cal+1)/cal_num)
        test_features = features[l_edge:r_edge,:]
        test_labels = labels[l_edge:r_edge,:]
        train_features = np.row_stack((features[0:l_edge,:],features[r_edge:sample_num,:]))
        train_labels = np.row_stack((labels[0:l_edge, :], labels[r_edge:sample_num, :]))
        predict_label = []
        # 9中体质类型分别运算
        for type in range(0,9):
            # svm训练
            module = svmutil.svm_train(train_labels[:,type].tolist(),
                                       train_features.tolist(),
                                       '')
            # svm验证
            result,_,_ = svmutil.svm_predict(test_labels[:,type].tolist(),
                                             test_features.tolist(),
                                             module,
                                             '')
            predict_label.append(result)
            # 输出进度提示
            print('order all:',len(orders),'cur:',order_num)
        predict_label = np.transpose(np.array(predict_label))
        # 统计在该测试集上的准确率
        # 参考 http://blog.csdn.net/bemachine/article/details/10471383 上的多标签分类评价标准
        acc = np.sum( np.sum(np.all([predict_label,test_labels],axis=0),axis=1) / np.sum(np.any([predict_label,test_labels],axis=0),axis=1) )/predict_label.shape[0]
        pre = np.sum(np.sum(np.all([predict_label,test_labels],axis=0), axis=1) / np.sum(predict_label, axis=1)) /predict_label.shape[0]
        rec = np.sum(np.sum(np.all([predict_label,test_labels],axis=0), axis=1) / np.sum(test_labels, axis=1)) /predict_label.shape[0]
        f = 2*pre*rec/(pre+rec)
        measures += [acc,pre,rec,f]
        wb.active.append([features_name] + [acc,pre,rec,f])
    # 求不同数据集测试结果平均值
    measures /= run_num
    # 保存结果到xmls文件
    # wb.active.append([features_name]+measures.tolist())
try:
    wb.save(r"C:\offline\体质辨识数据备份\origin\运行结果.xlsx")
except PermissionError:
    print("关闭excel文件继续")
    import dlib
    dlib.hit_enter_to_continue()
    wb.save(r"C:\offline\体质辨识数据备份\origin\运行结果.xlsx")


