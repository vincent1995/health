# 实现现有的论文中的特征提取方法，用作比较
import glob
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
from skimage import io
import numpy as np
####################################################
# 准备工作
####################################################
path =r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\4466"


####################################################
# 实现  论文中的提取方法
####################################################
# No.1 RGB空间下，面部颜色均值 (face_block_avg_color)
# 论文：李福凤
# 相关论文：毛红潮，傅言
def get_avg_color():
    try:
        f = open(path+'\\' +"face_block_avg_color", 'rb')
        avg_color=pickle.load(f)
    except Exception:
        avg_color = {}
    for q,f in enumerate(glob.glob(path+r"\face\*.jpg")):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if (ID in avg_color.keys()):
            continue
        # 获得特征
        img = io.imread(f)
        avg_color[ID] = np.mean(img,axis=(0,1)).tolist()
    f = open(path + '\\' + "face_block_avg_color", 'wb')
    pickle.dump(avg_color,f)

# No.2 通过fcm获得的皮肤主色
# 论文：吴敦华
def get_fcm_color():
    from feature_extract import fcm
    fcmProcess = fcm.FCM()
    try:
        f = open(path+'\\' +"face_fcm_color", 'rb')
        fcm_color=pickle.load(f)
    except Exception:
        fcm_color = {}
    for q, f in enumerate(glob.glob(path + r"\face\*.jpg")):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if (ID in fcm_color.keys()):
            continue
        # 获得特征
        img = io.imread(f)
        fcm_color[ID] = fcmProcess.run(img)
    f = open(path + '\\' + "face_fcm_color", 'wb')
    pickle.dump(fcm_color, f)

# No.2 通过fcm获得的皮肤主色
# 论文：吴敦华
def get_fcm_color():
    from feature_extract import fcm
    fcmProcess = fcm.FCM()
    try:
        f = open(path+'\\' +"face_fcm_color", 'rb')
        fcm_color=pickle.load(f)
    except Exception:
        fcm_color = {}
    for q, f in enumerate(glob.glob(path + r"\face\*.jpg")):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if (ID in fcm_color.keys()):
            continue
        # 获得特征
        img = io.imread(f)
        fcm_color[ID] = fcmProcess.run(img)
    f = open(path + '\\' + "face_fcm_color", 'wb')
    pickle.dump(fcm_color, f)

###################################################
# 运行
###################################################
get_avg_color()
get_fcm_color()