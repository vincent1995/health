#!/usr/bin/python
import dlib

predictor_path = "module.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def facelandmark(pic):
    dets = detector(pic, 1)
    rtn = []
    flag = False
    if(len(dets)>0):
        # 获得面部关键点
        points = predictor(pic, dets[0])
        flag = True
        # 提取左右面部皮肤块
        # 左脸
        p1 = points.part(36)
        p2 = points.part(31)
        p3 = dlib.point(int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))
        dis = ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**(1/2)
        rectl = dlib.rectangle(int(p3.x-dis*0.3),int(p3.y),int(p3.x),int(p3.y+dis*0.3))
        # 右脸
        p1 = points.part(45)
        p2 = points.part(35)
        p3 = dlib.point(int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))
        dis = ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**(1/2)
        rectr = dlib.rectangle(int(p3.x),int(p3.y),int(p3.x+dis*0.3),int(p3.y+dis*0.3))
        rtn = [dets[0],rectl,rectr]
    # 返回脸部范围，左脸皮肤块，右脸皮肤块
    return flag, rtn


def drawRect(pic,rects):
    from skimage import draw
    img = pic
    for rect in rects:
        rectl = dlib.rectangle(rect[2],rect[0],rect[3],rect[1])
        rr, cc = draw.line(rectl.top(), rectl.left(), rectl.bottom(), rectl.left())
        img[rr, cc, :] = (0, 255, 0)
        rr, cc = draw.line(rectl.bottom(), rectl.left(), rectl.bottom(), rectl.right())
        img[rr, cc, :] = (0, 255, 0)
        rr, cc = draw.line(rectl.bottom(), rectl.right(), rectl.top(), rectl.right())
        img[rr, cc, :] = (0, 255, 0)
        rr, cc = draw.line(rectl.top(), rectl.right(), rectl.top(), rectl.left())
        img[rr, cc, :] = (0, 255, 0)
    return img

# def get_face_rects():
#     # 导入包
#     import glob
#     from skimage import io
#     import os
#     try:
#         import cPickle as pickle
#     except ImportError:
#         import pickle
#     # 需要获得的数据
#     all_rects = []
#     flags = []
#     all_labels = []
#     # 提取皮肤块矩形
#     faces_folder_path = r"C:\Users\vincent\OneDrive\online\labeled photo"
#     for i,f in enumerate(glob.glob(os.path.join(faces_folder_path, "*.jpg"))):
#         img = io.imread(f)
#         # 脸部特征点定位
#         flag, rects = facelandmark(img)
#         flags.append(flag)
#         if flag:
#             all_rects.append(rects)
#         else:
#             all_rects.append(None)
#         # 获得样本标签
#         if flag:
#             # 获得样本标签
#             filename = os.path.split(f)[1]
#             filename = os.path.splitext(filename)[0]
#             fileLables = filename.split("_")[1:]
#             all_labels.append(fileLables)
#     # 保存矩形信息
#     f = open("landmark_data",'wb')
#     pickle.dump([flags,all_rects,all_labels],f)


# 使用数据库的landmark程序
def get_face_rects(path):
    # 导入包
    import glob
    from skimage import io
    import os
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import sys
    # import json
    # sys.path.append('../sql')
    # from sql import SQL
    # sql = SQL()
    # 需要获得的数据
    features = []
    # 提取皮肤块矩形
    faces_folder_path = path+r"\face"
    for i,f in enumerate(glob.glob(os.path.join(faces_folder_path, "*.jpg"))):
        img = io.imread(f)
        # 脸部特征点定位
        print(i,f)
        flag, rects = facelandmark(img)
        if flag:
            # 存入 数据库
            face = [rects[0].top(),rects[0].bottom(),rects[0].left(),rects[0].right()]
            l_skin = [rects[1].top(),rects[1].bottom(),rects[1].left(),rects[1].right()]
            r_skin = [rects[2].top(),rects[2].bottom(),rects[2].left(),rects[2].right()]
            d = {'face':face,'l_skin':l_skin,'r_skin':r_skin}
            features.append(d)
            # print(d)
            # fname = os.path.split(f)[1]
            # id = fname.split('_')[0]
            # sql.insert('origin_face_feature','landmark',id,d)
        else:
            features.append(None)
        # # 获得样本标签
        # if flag:
        #     # 获得样本标签
        #     filename = os.path.split(f)[1]
        #     filename = os.path.splitext(filename)[0]
        #     fileLables = filename.split("_")[1:]
        #     all_labels.append(fileLables)
    f = open(path+r"\face_landmark",'wb')
    pickle.dump(features,f)


if __name__ == "__main__":
    import os
    import glob
    from skimage import io
    # 显示图片
    # for f in glob.glob(r"C:\offline\体质辨识数据备份\origin\face\*.jpg"):
    #     img = io.imread(f)
    #     print(type(img))
    #     flag, rects = facelandmark(img)
    #     if flag:
    #         img = drawRect(img,rects)
    #         io.imshow(img)
    #         print(type(rects[0]))
    #         new_img = img[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right(),:]
    #         print(new_img.shape)
    #         io.imshow(new_img)
    #         io.show()
    #     dlib.hit_enter_to_continue()

    # 获得landmark
    path =r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin"
    get_face_rects(path)

    # 显示图片
    # try:
    #     import cPickle as pickle
    # except ImportError:
    #     import pickle
    # f = open(path+r"\face_landmark", 'rb')
    # data = pickle.load(f)
    # for i in data:
    #     print(i)
    # for d,f in zip(data,glob.glob(r"C:\offline\体质辨识数据备份\origin\face\*.jpg")):
    #     if not d == None:
    #         rect = d['face']
    #         a = abs(rect[0]-rect[1])*abs(rect[2]-rect[3])
    #         area.append(a)
    #         if a >80000 and a<90000:
    #             print(a)
    #             img = io.imread(f)
    #             try:
    #                 drawRect(img,[rect])
    #             except Exception:
    #                 pass
    #             io.imshow(img)
    #             io.show()
    #             dlib.hit_enter_to_continue()
    # import matplotlib.pyplot as plt
    # plt.hist(area,bins='auto')
    # plt.show()