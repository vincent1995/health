import numpy as np
import glob
from skimage import io, color, feature, img_as_float, img_as_ubyte, transform
from numpy import *
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os


def LBP(im):
    Method = 'uniform'
    P = 8
    R = 3
    im_gray = color.rgb2gray(im)
    im_gray = img_as_ubyte(im_gray)
    lbp = np.zeros(10, dtype=np.float)
    l = feature.local_binary_pattern(im_gray, P, R, Method)
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            lbp[int(l[i, j])] += 1
    sum_lbp = np.sum(lbp)
    lbp[:] = lbp[:] / sum_lbp
    return lbp


def rgb2hsv(arr):
    out = np.empty_like(arr)
    # -- V channel
    out_v = arr.max(-1)
    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.
    # -- H channel
    # red is max
    idx = (arr[:, :, 0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:, :, 1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = (arr[:, :, 2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)
    out_h[:] = np.round(out_h[:] * 360)
    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v
    # remove NaN
    out[np.isnan(out)] = 0
    return out


def rank_hsv(im):  # 为HSV色彩空间进行量化，12个H分量级，3个S分量,1个V分量
    L = im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if 0 < L[i, j, 0] <= 20:
                L[i, j, 0] = 0
            elif 20 < L[i, j, 0] <= 75:
                L[i, j, 0] = 1
            elif 75 < L[i, j, 0] <= 155:
                L[i, j, 0] = 2
            elif 155 < L[i, j, 0] <= 190:
                L[i, j, 0] = 3
            elif 190 < L[i, j, 0] <= 270:
                L[i, j, 0] = 4
            elif 270 < L[i, j, 0] <= 295:
                L[i, j, 0] = 5
            elif 295 < L[i, j, 0] <= 310.0:
                L[i, j, 0] = 6
            elif 310 < L[i, j, 0] <= 320.0:
                L[i, j, 0] = 7
            elif 320 < L[i, j, 0] <= 330.0:
                L[i, j, 0] = 8
            elif 330 < L[i, j, 0] <= 340.0:
                L[i, j, 0] = 9
            elif 340 < L[i, j, 0] <= 350.0:
                L[i, j, 0] = 10
            else:
                L[i, j, 0] = 11

            if 0 <= L[i, j, 1] < 0.2:
                L[i, j, 1] = 0
            elif 0.2 <= L[i, j, 1] < 0.7:
                L[i, j, 1] = 1
            else:
                L[i, j, 1] = 2

            L[i, j, 2] = 0  # V分量为亮度，因此均将其量化为1不进行考虑
    return L


def Hist_hsv(im):  # 得到量化过后36柄的一维直方图(全局颜色直方图)
    H = np.zeros(36, float)
    position = (3 * im[..., 0] + im[..., 1])
    for i in range(position.shape[0]):
        for j in range(position.shape[1]):
            H[int(position[i, j])] += 1
    Sum_H = sum(H)
    H[:] = H[:] / Sum_H
    return H


def getColorRec(im_hsv):  # 计算HSV空间中H和S通道的颜色距（均值，标准差，三阶矩）一共6个分量
    Whole_Color = []
    m = im_hsv.shape[0]
    n = im_hsv.shape[1]
    for i in range(2):
        t = im_hsv[:, :, i]
        mean = np.float64(np.mean(t))
        std = np.float64(np.std(t))
        three = np.sum((t - mean) ** 3)
        three /= m * n
        if three > 0:
            three = np.power(three, 1 / 3)
        else:
            three = -(np.power(-three, 1 / 3))
        three = np.float64(three)
        Whole_Color += [mean, std, three]
    return Whole_Color


# 基于傅里叶变换的纹理能量提取
# 采用长方环周向谱能量百分比法
# 均匀的将图像功率谱分成M=20个等宽度的长方环，求出每一个长方环功率谱能量占总能量的百分比
# 长方环内的功率谱能量可以反映出图像不同频率成分的能量强度
def get_power(F):
    m = F.shape[0]
    n = F.shape[1]
    M = 20
    central_x = m / 2  # 图像中心坐标值
    central_y = n / 2
    P = np.zeros([m, n], float)
    fh = np.zeros(20, float)
    P = F[...].real ** 2 + F[...].imag ** 2
    sum_P = np.sum(P)
    # for i in range (m):
    #     for j in range (n):
    #         real=F[i,j].real
    #         imag=F[i,j].imag
    #         P[i,j]=pow(F[i,j].real,2)+pow(F[i,j].imag,2)
    # sum_P=sum(sum(P))
    for i in range(m):
        for j in range(n):
            for k in range(20):
                if (m * k) / (2 * M) <= abs(i - central_x) < m * (k + 1) / (2 * M) and (n * k) / (2 * M) <= abs(
                                j - central_y) < n * (k + 1) / (2 * M):
                    fh[k] += P[i, j]
    for i in range(size(fh)):
        fh[i] /= sum_P
    return fh


# 基于共生矩阵纹理特征提取,计算四个共生矩阵P，取距离为dist，灰度级为G_class(一般取8或者16)角度分别为0，45,90,135
# 返回八维纹理特征行向量
def Gray_Texture(im, dist, G_class):  # im为灰度图像
    T = np.zeros(8, float)
    im = img_as_ubyte(im)
    gcm = feature.greycomatrix(im, dist, [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], normed=True)
    T[0] = np.mean(feature.greycoprops(gcm, 'energy'))
    T[1] = np.std(feature.greycoprops(gcm, 'energy'))
    T[2] = np.mean(feature.greycoprops(gcm, 'contrast'))
    T[3] = np.std(feature.greycoprops(gcm, 'contrast'))
    T[4] = np.mean(feature.greycoprops(gcm, 'dissimilarity'))
    T[5] = np.std(feature.greycoprops(gcm, 'dissimilarity'))
    T[6] = np.mean(feature.greycoprops(gcm, 'correlation'))
    T[7] = np.std(feature.greycoprops(gcm, 'correlation'))
    return T
    # m=im.shape[0]
    # n=im.shape[1]
    # Gray=np.zeros([m,n],int)
    # #原始图像灰度级压缩，将Gray量化成16级
    # for i in range(m):
    #     for j in range(n):
    #         for k in range(G_class):
    #             if k*G_class<=im[i,j]<=(k*G_class)+(G_class-1):
    #                 Gray[i,j]=k
    #
    # P=np.zeros([G_class,G_class,4],float)
    #
    # for i in range(m):
    #     for j in range(n):
    #         if j < n-dist:
    #             P[Gray[i,j],Gray[i,j+dist],0]+=1
    #         if i>=dist and j<n-dist:
    #             P[Gray[i,j],Gray[i-dist,j+dist],1]+=1
    #         if i<m-dist:
    #             P[Gray[i,j],Gray[i+dist,j],2]+=1
    #         if i<m-dist and j<n-dist:
    #             P[Gray[i,j],Gray[i+dist,j+dist],3]+=1
    # ##对灰度共生矩阵进行归一化
    # for i in range(4):
    #     p_sum=sum(sum(P[:,:,i]))
    #     P[:,:,i]=P[:,:,i]/p_sum
    # H = np.zeros(4, float)
    # I = np.zeros(4, float)
    # Ux = np.zeros(4, float)
    # Uy = np.zeros(4, float)
    # E = np.zeros(4, float)
    # deltaX = np.zeros(4, float)
    # deltaY = np.zeros(4, float)
    # C = np.zeros(4, float)
    # T = np.zeros(8, float)  # 存储能量、熵、惯性矩、相关的均值和标准差作为最终8维纹理特征
    # # E = np.sum(P**2,axis=(0,1))
    # # Ux = np.sum(np.dot(range(G_class), P),axis=(0,1))
    # # Uy = np.sum(np.dot( P,range(G_class)), axis=(0, 1))
    # for l in range(4):
    #     for i in range(G_class):
    #         for j in range(G_class):
    #             E[l] += pow(P[i, j, l], 2)  # 能量
    #             if P[i, j, l] > 0:
    #                 H[l] = H[l] - P[i, j, l] * math.log2(P[i, j, l])  # 熵
    #             I[l] = pow((i - j), 2) * P[i, j, l] + I[l]  # 惯性矩（二阶距）
    #             Ux[l] = i * P[i, j, l] + Ux[l]  # 相关性中的μx
    #             Uy[l] = j * P[i, j, l] + Uy[l]  # 相关性中的μy
    #
    # for l in range(4):
    #     for i in range(G_class):
    #         for j in range(G_class):
    #             deltaX[l] += pow(i - Ux[l], 2) * P[i, j, l]
    #             deltaY[l] += pow(j - Uy[l], 2) * P[i, j, l]
    #             C[l] += i * j * P[i, j, l]
    #
    #     C[l] = (C[l] - Ux[l] * Uy[l]) / deltaX[l] / deltaY[l]  # 相关性
    # T[0] = np.mean(E)
    # T[1] = np.std(E)
    # T[2] = np.mean(H)
    # T[3] = np.std(H)
    # T[4] = np.mean(I)
    # T[5] = np.std(I)
    # T[6] = np.mean(C)
    # T[7] = np.std(C)
    # return T


def get_tongue_feature(path):
    part = 'small_tongue'
    # len=len(glob.glob(r"C:\offline\体质辨识数据备份\origin\cutted_tongue\*.jpg"))#图像的总个数
    # 需要的图像的属性：36+6个颜色特征，38个纹理特征(20个傅里叶能量，10个LBP，8个灰度共生矩阵)
    try:
        f = open(path+'\\' + part + '_color_hist', 'rb')
        color_hist=pickle.load(f)
    except Exception:
        color_hist = {}
    try:
        f = open(path+'\\' + part + '_color_rect', 'rb')
        color_rect=pickle.load(f)
    except Exception:
        color_rect={}
    try:
        f = open(path+'\\' + part + '_gray_texture', 'rb')
        gray_texture=pickle.load(f)
    except Exception:
        gray_texture = {}
    try:
        f = open(path+'\\' + part + '_power', 'rb')
        power=pickle.load(f)
    except Exception:
        power = {}
    try:
        f = open(path+'\\' + part + '_lbp', 'rb')
        lbp=pickle.load(f)
    except Exception:
        lbp = {}
    for q, f in enumerate(glob.glob(path+r"\cutted_tongue\*.jpg")):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if(ID in color_hist.keys()):
            continue
        # 计算特征
        img = io.imread(f)
        # 排除太小的图片
        if(img.shape[0]*img.shape[1]<20000):
            continue
        img = transform.resize(img, (128, 128))
        im_hsv = rgb2hsv(img_as_float(img))  # 将图像转到HSV色彩空间
        L = rank_hsv(im_hsv)  # 量化HSV色彩空间
        H = Hist_hsv(L)  # 求HSV全局直方图
        Whole_Color = getColorRec(im_hsv)  # HSV通道H通道（色调）和S通道（饱和度）的颜色距
        img_gray = color.rgb2gray(img)
        f1 = np.fft.fft2(img_gray)  # 二维离散傅里叶变换
        fshift = np.fft.fftshift(f1)
        fh = get_power(fshift)  # 均匀的将图像功率谱分成M=20个等宽度的长方环，求出每一个长方环功率谱能量占总能量的百分比
        T = Gray_Texture(img_gray, [1], 16)  # 灰度共生矩阵得到8维特征
        one_lbp = LBP(img)
        # 存储特征
        color_hist[ID]=H.tolist()
        color_rect[ID]=Whole_Color
        gray_texture[ID] =T.tolist()
        power[ID]=fh.tolist()
        lbp[ID]=one_lbp.tolist()
        # for i in range(36): #得到HSV的全局色彩直方图
        #     dataMat[q,i]=H[i]
        # num=0
        # for i in range(36,42):
        #     dataMat[q,i]=Whole_Color[num]
        #     num+=1
        # num=0
        # for i in range(42,50):#得到灰度共生矩阵
        #     dataMat[q,i]=T[num]
        #     num+=1
        # num=0
        # for i in range(50,70):#得到傅里叶能量谱
        #     dataMat[q,i]=fh[num]
        #     num+=1
        # num=0
        # for i in range(70,80): #得到LBP算子
        #     dataMat[q,i]=lbp[num]
        #     num+=1
    # 保存特征
    f = open(path+'\\' + part + '_color_hist', 'wb')
    pickle.dump(color_hist, f)
    f = open(path+'\\' + part + '_color_rect', 'wb')
    pickle.dump(color_rect, f)
    f = open(path+'\\' + part + '_gray_texture', 'wb')
    pickle.dump(gray_texture, f)
    f = open(path+'\\' + part + '_power', 'wb')
    pickle.dump(power, f)
    f = open(path+'\\' + part + '_lbp', 'wb')
    pickle.dump(lbp, f)

    # np.save(r"C:\offline\体质辨识数据备份\origin\tongue_feature",dataMat)
    # np.save('labelMat1245_test.npy',labelMat)


def get_face_block_feature(path):
    part = 'face_block'
    # len=len(glob.glob(r"C:\offline\体质辨识数据备份\origin\cutted_tongue\*.jpg"))#图像的总个数
    # 需要的图像的属性：36+6个颜色特征，38个纹理特征(20个傅里叶能量，10个LBP，8个灰度共生矩阵)
    try:
        f = open(path+'\\' + part + '_color_hist', 'rb')
        color_hist=pickle.load(f)
    except Exception:
        color_hist = {}
    try:
        f = open(path+'\\' + part + '_color_rect', 'rb')
        color_rect=pickle.load(f)
    except Exception:
        color_rect={}
    try:
        f = open(path+'\\' + part + '_gray_texture', 'rb')
        gray_texture=pickle.load(f)
    except Exception:
        gray_texture = {}
    try:
        f = open(path+'\\' + part + '_power', 'rb')
        power=pickle.load(f)
    except Exception:
        power = {}
    try:
        f = open(path+'\\' + part + '_lbp', 'rb')
        lbp=pickle.load(f)
    except Exception:
        lbp = {}
    # 载入数据
    f = open(path+r"\face_landmark", 'rb')
    landmarks = pickle.load(f)
    for q, (f, landmark) in enumerate(zip(glob.glob(path+r"\face\*.jpg"), landmarks)):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if(ID in color_hist.keys()):
            continue
        # 获得特征
        if landmark == None:
            continue
        else:
            img = io.imread(f)
            l_skin, r_skin = landmark['l_skin'], landmark['r_skin']
            l_skin[0], l_skin[2] = max(l_skin[0], 0), max(l_skin[2], 0)
            l_skin[1], l_skin[3] = min(l_skin[1], img.shape[0]), min(l_skin[3], img.shape[1])
            r_skin[0], r_skin[2] = max(r_skin[0], 0), max(r_skin[2], 0)
            r_skin[1], r_skin[3] = min(r_skin[1], img.shape[0]), min(r_skin[3], img.shape[1])
            # 皮肤块太小不进行提取
            if abs(l_skin[0] - l_skin[1]) * abs(l_skin[2] - l_skin[3]) < 400 or \
                                    abs(r_skin[0] - r_skin[1]) * abs(r_skin[2] - r_skin[3]) < 400:
                continue
            else:
                # 计算左脸
                leftFace = img[l_skin[0]:l_skin[1], l_skin[2]:l_skin[3]].copy()
                leftFace = transform.resize(leftFace, (60, 60))
                l_hsv = rgb2hsv(img_as_float(leftFace))  # 将图像转到HSV色彩空间
                l_H = Hist_hsv(rank_hsv(l_hsv))  # 求HSV全局直方图
                l_Whole_Color = getColorRec(l_hsv)  # HSV通道H通道（色调）和S通道（饱和度）的颜色距
                l_gray = color.rgb2gray(leftFace)
                l_fh = get_power(np.fft.fftshift(np.fft.fft2(l_gray)))  # 均匀的将图像功率谱分成M=20个等宽度的长方环，求出每一个长方环功率谱能量占总能量的百分比
                l_T = Gray_Texture(l_gray, [1], 16)  # 灰度共生矩阵得到8维特征
                l_lbp = LBP(leftFace)
                # 计算右脸
                rightFace = img[r_skin[0]:r_skin[1], r_skin[2]:r_skin[3]].copy()
                rightFace = transform.resize(rightFace, (60, 60))
                r_hsv = rgb2hsv(img_as_float(rightFace))  # 将图像转到HSV色彩空间
                r_H = Hist_hsv(rank_hsv(r_hsv))  # 求HSV全局直方图
                r_Whole_Color = getColorRec(r_hsv)  # HSV通道H通道（色调）和S通道（饱和度）的颜色距
                r_gray = color.rgb2gray(rightFace)
                r_fh = get_power(np.fft.fftshift(np.fft.fft2(r_gray)))  # 均匀的将图像功率谱分成M=20个等宽度的长方环，求出每一个长方环功率谱能量占总能量的百分比
                r_T = Gray_Texture(r_gray, [1], 16)  # 灰度共生矩阵得到8维特征
                r_lbp = LBP(rightFace)
                # 整合特征
                H = (l_H + r_H) / 2
                Whole_Color = (np.array(l_Whole_Color) + np.array(r_Whole_Color)) / 2
                T = (l_T + r_T) / 2
                fh = (l_fh + r_fh) / 2
                one_lbp = (l_lbp + r_lbp)
                # 存储特征
                color_hist[ID] = H.tolist()
                color_rect[ID] = Whole_Color.tolist()
                gray_texture[ID] = T.tolist()
                power[ID] = fh.tolist()
                lbp[ID] = one_lbp.tolist()

    # 保存特征
    f = open(path+'\\' + part + '_color_hist', 'wb')
    pickle.dump(color_hist, f)
    f = open(path+'\\' + part + '_color_rect', 'wb')
    pickle.dump(color_rect, f)
    f = open(path+'\\' + part + '_gray_texture', 'wb')
    pickle.dump(gray_texture, f)
    f = open(path+'\\' + part + '_power', 'wb')
    pickle.dump(power, f)
    f = open(path+'\\' + part + '_lbp', 'wb')
    pickle.dump(lbp, f)

def get_face_feature(path):
    part = 'face'
    # len=len(glob.glob(r"C:\offline\体质辨识数据备份\origin\cutted_tongue\*.jpg"))#图像的总个数
    # 需要的图像的属性：36+6个颜色特征，38个纹理特征(20个傅里叶能量，10个LBP，8个灰度共生矩阵)
    try:
        f = open(path+'\\' + part + '_color_hist', 'rb')
        color_hist=pickle.load(f)
    except Exception:
        color_hist = {}
    try:
        f = open(path+'\\' + part + '_color_rect', 'rb')
        color_rect=pickle.load(f)
    except Exception:
        color_rect={}
    try:
        f = open(path+'\\' + part + '_gray_texture', 'rb')
        gray_texture=pickle.load(f)
    except Exception:
        gray_texture = {}
    try:
        f = open(path+'\\' + part + '_power', 'rb')
        power=pickle.load(f)
    except Exception:
        power = {}
    try:
        f = open(path+'\\' + part + '_lbp', 'rb')
        lbp=pickle.load(f)
    except Exception:
        lbp = {}
    # 载入数据
    f = open(path+r"\face_landmark", 'rb')
    landmarks = pickle.load(f)
    for q, (f, landmark) in enumerate(zip(glob.glob(path+r"\face\*.jpg"), landmarks)):
        # 获得患者ID
        ID = os.path.split(f)[1].split('.')[0].split('_')[0]
        print(q, ID)
        if(ID in color_hist.keys()):
            continue
        # 获得特征
        if landmark == None:
            continue
        else:
            img = io.imread(f)
            face = landmark['face']
            face[0], face[2] = max(face[0], 0), max(face[2], 0)
            face[1], face[3] = min(face[1], img.shape[0]), min(face[3], img.shape[1])
            #  脸部太小不进行提取
            if abs(face[0] - face[1]) * abs(face[2] - face[3]) < 100000:
                continue
            else:
                # print(q)
                img = img[face[0]:face[1], face[2]:face[3]].copy()
                img = transform.resize(img, (256, 256))
                im_hsv = rgb2hsv(img_as_float(img))  # 将图像转到HSV色彩空间
                L = rank_hsv(im_hsv)  # 量化HSV色彩空间
                H = Hist_hsv(L)  # 求HSV全局直方图
                Whole_Color = getColorRec(im_hsv)  # HSV通道H通道（色调）和S通道（饱和度）的颜色距
                img_gray = color.rgb2gray(img)
                f1 = np.fft.fft2(img_gray)  # 二维离散傅里叶变换
                fshift = np.fft.fftshift(f1)
                fh = get_power(fshift)  # 均匀的将图像功率谱分成M=20个等宽度的长方环，求出每一个长方环功率谱能量占总能量的百分比
                T = Gray_Texture(img_gray, [1], 16)  # 灰度共生矩阵得到8维特征
                one_lbp = LBP(img)
                # 存储特征
                color_hist[ID] = H.tolist()
                color_rect[ID] = Whole_Color
                gray_texture[ID] = T.tolist()
                power[ID] = fh.tolist()
                lbp[ID] = one_lbp.tolist()

    # 保存特征
    f = open(path+'\\' + part + '_color_hist', 'wb')
    pickle.dump(color_hist, f)
    f = open(path+'\\' + part + '_color_rect', 'wb')
    pickle.dump(color_rect, f)
    f = open(path+'\\' + part + '_gray_texture', 'wb')
    pickle.dump(gray_texture, f)
    f = open(path+'\\' + part + '_power', 'wb')
    pickle.dump(power, f)
    f = open(path+'\\' + part + '_lbp', 'wb')
    pickle.dump(lbp, f)


if __name__ == '__main__':
    path = r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin"
    # get_tongue_feature(path)
    # get_face_block_feature(path)
    get_face_feature(path)
