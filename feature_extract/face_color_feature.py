
def get_face_color_feature(path):
    import glob
    from skimage import io, transform
    from feature_extract import fcm
    import datetime as time
    import os
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    # 载入数据
    f = open(path+r"\face_landmark","rb")
    landmark_datas = pickle.load(f)
    # 需要获得结果
    features = {}
    # 运行程序
    faces_folder_path = path+r"\face"
    fcmProcess = fcm.FCM()
    t1 = time.datetime.now()
    print("start feature extract")
    for i,(landmark,f) in enumerate(zip(landmark_datas,glob.glob(os.path.join(faces_folder_path, "*.jpg")))):
        if not landmark==None:
            img = io.imread(f)
            l_skin,r_skin = landmark['l_skin'],landmark['r_skin']
            l_skin[0], l_skin[2] = max(l_skin[0], 0), max(l_skin[2], 0)
            l_skin[1], l_skin[3] = min(l_skin[1], img.shape[0]), min(l_skin[3], img.shape[1])
            r_skin[0], r_skin[2] = max(r_skin[0], 0), max(r_skin[2], 0)
            r_skin[1], r_skin[3] = min(r_skin[1], img.shape[0]), min(r_skin[3], img.shape[1])
            # 皮肤块太小不进行提取
            if abs(l_skin[0] - l_skin[1]) * abs(l_skin[2] - l_skin[3]) < 400 or \
                                    abs(r_skin[0] - r_skin[1]) * abs(r_skin[2] - r_skin[3]) < 400:
                continue
            else:
                # 提取左脸皮肤块
                leftFace = img[l_skin[0]:l_skin[1],l_skin[2]:l_skin[3]].copy()
                leftFace = transform.resize(leftFace,(60,60))
                # 运行fcm算法
                leftColor = fcmProcess.run(leftFace)
                # 提取右脸皮肤块
                rightFace = img[r_skin[0]:r_skin[1],r_skin[2]:r_skin[3]].copy()
                rightFace = transform.resize(rightFace, (60, 60))
                # 运行fcm算法
                rightColor = fcmProcess.run(rightFace)
                color = leftColor/2 + rightColor/2 # 得到最后的面色特征

                # 获得患者ID
                ID = os.path.split(f)[1].split('.')[0].split('_')[0]
                print(i, ID,color)
                # 保存数据
                features[ID]=color.tolist()


    t2 = time.datetime.now()
    print("feature extract time: ", (t2-t1).seconds)

    # 保存特征信息
    f = open(path+r"\face_main_color",'wb')
    pickle.dump(features,f)

if __name__ == '__main__':
    path = r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin"
    # try:
    #     import cPickle as pickle
    # except ImportError:
    #     import pickle
    # f = open(r"c:\offline\体质辨识数据备份\origin\face_main_color",'rb')
    # features = pickle.load(f)
    # print(features)
    get_face_color_feature(path)
    # changeListToDict(path,feature)