from skimage import feature,color,transform
import numpy as np
METHOD = 'uniform'
radius = 3
n_points = 8

def local_binary_feature(pic):
    gray_pic = color.rgb2gray(pic)
    lbp = feature.local_binary_pattern(gray_pic,n_points,radius,METHOD)
    rtn = np.zeros(n_points+2)
    for row in lbp:
        for pin in row:
            rtn[int(pin)]+=1
    return rtn

# 获得lbp 特征
def get_lbp_feature():
    # 导入包
    import glob
    from skimage import io
    import os
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    # 载入数据
    f = open(r"c:\offline\体质辨识数据备份\origin\face_landmark","rb")
    landmark_datas = pickle.load(f)
    # 需要获得结果
    features = []
    # 运行程序
    faces_folder_path = r"c:\offline\体质辨识数据备份\origin\face"
    print("start feature extract")
    for i,(landmark,f) in enumerate(zip(landmark_datas,glob.glob(os.path.join(faces_folder_path, "*.jpg")))):
        if not landmark==None:
            img = io.imread(f)
            l_skin,r_skin = (landmark['l_skin'],landmark['r_skin'])
            l_skin[0],l_skin[2] = max(l_skin[0],0),max(l_skin[2],0)
            l_skin[1],l_skin[3] = min(l_skin[1],img.shape[0]),min(l_skin[3],img.shape[1])
            r_skin[0],r_skin[2] = max(r_skin[0],0),max(r_skin[2],0)
            r_skin[1],r_skin[3] = min(r_skin[1],img.shape[0]),min(r_skin[3],img.shape[1])
            # 皮肤块太小不进行提取
            if abs(l_skin[0] - l_skin[1]) * abs(l_skin[2] - l_skin[3]) < 400 or \
                abs(r_skin[0] - r_skin[1]) * abs(r_skin[2] - r_skin[3]) < 400:
                features.append(None)
            else:
                # 提取左脸皮肤块
                leftFace = img[l_skin[0]:l_skin[1],l_skin[2]:l_skin[3]].copy()
                leftFace = transform.resize(leftFace,(60,60))
                # 运行lbp算法
                lbpl = local_binary_feature(leftFace)
                # 提取右脸皮肤块
                rightFace = img[r_skin[0]:r_skin[1],r_skin[2]:r_skin[3]].copy()
                rightFace = transform.resize(rightFace, (60, 60))
                # 运行lbp算法
                lbpr = local_binary_feature(rightFace)
                # 获得患者ID
                ID = os.path.split(f)[1].split('_')[0]
                print(i, ID,lbpl,lbpr)
                features.append([ID]+np.hstack((lbpl, lbpr)).tolist())
        else:
            features.append(None)

    # 保存特征信息
    f = open(r"c:\offline\体质辨识数据备份\origin\face_small_area_lbp_feature", 'wb')
    pickle.dump(features, f)

if __name__ == '__main__':
    # from skimage import io
    # img = io.imread(r"C:\Users\vincent\OneDrive\online\labeled photo\05202031273613_7_4.jpg")
    # feature = local_binary_feature(img)
    # print(feature)

    get_lbp_feature()

    # import pickle
    # f = open("lbp_feature",'rb')
    # data = pickle.load(f)
    # print(data[0])
    # print(data[199])
    # print(len(data))
