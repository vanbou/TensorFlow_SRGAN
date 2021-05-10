from glob import glob
import numpy as np
import scipy.misc


# @author: resumebb
# @time: 2020/11/20

# 数据加载类

class dataProcess():
    def __init__(self, data_name, img=(128, 128)):
        self.data_name = data_name
        self.img = img

    def process(self, batch_size=1, data_type=False):
        dataPath = glob('./datasets/%s/train/*' % (self.data_name))
        images = np.random.choice(dataPath, size=batch_size)

        img_HR = []
        img_LR = []
        for i in images:
            img = self.imread(i)
            height, width = self.img
            # 缩小4倍
            L_height, L_width = int(height / 4), int(width / 4)

            img_H = scipy.misc.imresize(img, self.img)
            img_L = scipy.misc.imresize(img, (L_height, L_width))

            if not data_type and np.random.random() < 0.5:
                img_H = np.fliplr(img_H)
                img_L = np.fliplr(img_L)

            img_HR.append(img_H)
            img_LR.append(img_L)

        img_HR = np.array(img_HR) / 127.5 - 1
        img_LR = np.array(img_LR) / 127.5 - 1

        return img_HR, img_LR

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
