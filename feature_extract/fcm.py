import numpy as np
from skimage import color

class FCM:
    ###############################
    # 变量声明
    def __init__(self):
        self.numClusters = 3
        self.maxIterations = 20
        self.fuzziness = 2
        self.epsilon = 2


    def init(self, pic):
        self.img = color.rgb2lab(pic)
        self.clusterCenters = np.zeros(
            (self.numClusters,
             self.img.shape[2]),
            dtype=np.float
        )
        self.clusterPixelNumber = np.zeros(self.numClusters, dtype=np.uint16)
        # initial membership
        self.membership = np.random.random((self.img.shape[0], self.img.shape[1], self.numClusters))
        s = np.sum(self.membership,axis=2)
        s.shape=(list(s.shape)+[1])
        self.membership = self.membership / s


    #####################################



    def run(self, pic):
        """
        运行fcm函数
        :param pic: 皮肤块（numpy数组形式）
        :return: 皮肤块主色lab值
        """
        self.init(pic)
        # Calculate the initial objective function just for kicks.
        lastJ = self.calculateObjectiveFunction()
        # 主要的迭代过程
        for iteration in range(0, self.maxIterations):
            # Calculate cluster centers from MFs.
            self.calculateClusterCentersFromMFs()
            # Then calculate the MFs from the cluster centers !
            self.calculateMFsFromClusterCenters()
            # Then see how our objective function is going.
            j = self.calculateObjectiveFunction()
            if abs(lastJ - j) < self.epsilon:
                break
            lastJ = j
        # 统计包含最多像素的聚簇
        index = np.argmax(self.membership,axis=2)
        for row in index:
            for pix in row:
                self.clusterPixelNumber[pix]+=1
        # 返回包含像素最多的聚簇颜色
        index = np.argmax(self.clusterPixelNumber)
        return self.clusterCenters[index]

    def calculateClusterCentersFromMFs(self):
        for c in range(self.numClusters):
            top = np.sum((self.membership[:,:, c] ** self.fuzziness).reshape(list(self.membership.shape[:2])+[1]) * self.img,axis=(0,1))
            bottom = np.sum((self.membership[:,:,c] ** self.fuzziness),axis=(0,1))
            self.clusterCenters[c,] = top / bottom

    def calculateMFsFromClusterCenters(self):
        dis = []
        for i in range(self.numClusters):
            dis.append(self.calcDistance(self.clusterCenters[i],self.img))
        for c in range(self.numClusters):
            top = dis[c]
            sumTerms = np.zeros(self.membership.shape[:2],dtype=np.float)
            for ck in range(self.numClusters):
                thisDistance = dis[ck]
                sumTerms += ((top / thisDistance) ** (2 / (self.fuzziness - 1)))
            self.membership[:,:,c] = (1 / sumTerms)

    def calculateObjectiveFunction(self):
        j = 0
        for c in range(self.numClusters):
            dis = self.calcDistance(self.clusterCenters[c],self.img)
            j += np.sum(dis * (self.membership[:,:,c] ** self.fuzziness))
        return j

    def calcDistance(self,center, pic):
        return np.sum((pic-center)**2,axis=2)**(1/2)


    ###########################################
    # 颜色空间转换
    # def RGBtoLAB(self, pix):
    #     r = self.companding(pix[0] / 255) * 100
    #     g = self.companding(pix[1] / 255) * 100
    #     b = self.companding(pix[2] / 255) * 100
    #     X = r * 0.4124 + g * 0.3576 + b * 0.1805
    #     Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    #     Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    #     # XYZ to LAB
    #     # divide white reference
    #     X = self.pivotXYZ(X / 95.047)
    #     Y = self.pivotXYZ(Y / 100.000)
    #     Z = self.pivotXYZ(Z / 108.883)
    #     pix[0] = 116 * Y - 16
    #     pix[1] = 500 * (X - Y)
    #     pix[2] = 200 * (Y - Z)
    #
    # def companding(self,c):
    #     if (c > 0.04045):
    #         return ((c + 0.055) / 1.055) ** 2.4
    #     return (c / 12.92)
    #
    # def pivotXYZ(self,c):
    #     if (c > 0.008856):
    #         return c ** (1.0 / 3.0)
    #     return (903.3 * c + 16) / 116
    #
    # def LABtoRGB(self,src):
    #     y = (src[0] + 16.0) / 116.0
    #     x = src[1] / 500.0 + y
    #     z = y - src[2] / 200.0
    #     white = np.array((95.047,100.000,108.883))
    #
    #     if x**3 > 0.008856:
    #         x = x**3
    #     else:
    #         x = (x - 16/116)/7.787
    #     x *= white[0]
    #
    #     if src[0] > (0.008856 * 903.3):
    #         y = ((src[0]+16)/116)**3
    #     else:
    #         y = src[0] / 903.3
    #     y *= white[1]
    #
    #     if z**3 > 0.008856:
    #         z = z**3
    #     else:
    #         z = (z - 16 / 116) / 7.787
    #     z *= white[2]
    #
    #
    #     x /= 100
    #     y /= 100
    #     z /= 100
    #     r = x * 3.2406 + y * -1.5372 + z * -0.4986
    #     g = x * -0.9689 + y * 1.8758 + z * 0.0415
    #     b = x * 0.0557 + y * -0.2040 + z * 1.0570
    #     src[0] = self.ToRGB(r)
    #     src[1] = self.ToRGB(g)
    #     src[2] = self.ToRGB(b)
    # def ToRGB(self,n):
    #     if n> 0.0031308:
    #         n = 1.055*(n**(1/2.4))-0.055
    #     else:
    #         n = 12.92*n
    #     n = 255*n
    #     if n <0:
    #         return 0
    #     elif n >255:
    #         return 255
    #     else:
    #         return n
        #########################################

if __name__ == "__main__":
    import os
    import glob
    from skimage import io,img_as_ubyte
    from feature_extract import facelandmark

    faces_folder_path = r"C:\Users\vincent\OneDrive\online\labeled photo"
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        img = io.imread(f)
        # 定位
        flag, rects = facelandmark.facelandmark(img)
        if flag:
            rect = rects[0]
            # 提取左脸皮肤块
            leftFace = img[rect.top():rect.bottom(),rect.left():rect.right()].copy()
            # 运行fcm算法
            fcm = FCM()
            main_color = fcm.run(leftFace)
            print(main_color)
            [[main_color]] = img_as_ubyte(color.lab2rgb([[main_color]])) # 转换颜色空间
            print(main_color)
            # 绘制图像
            from skimage import draw
            import dlib
            win1 = dlib.image_window()
            pic = np.zeros((img.shape[0],img.shape[1]+100,3),dtype=np.uint8)
            pic[:, :img.shape[1], :] = img
            # pic[:,:img.shape[1],:] = facelandmark.drawRect(img, rects)# 绘制带框的脸部图像
            rr,cc = draw.circle(50,img.shape[1]+50,40,pic.shape)
            pic[rr,cc,:] = main_color # 绘制面部主色
            win1.set_image(pic)
            dlib.hit_enter_to_continue()
    # pix = np.array((123,234,242))
    # fcm = FCM()
    # fcm.RGBtoLAB(pix)
    # print(pix)
    # fcm.LABtoRGB(pix)
    # print(pix)