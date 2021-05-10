import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import Model,Input
from keras.applications import VGG19
from keras.layers import Conv2D, BatchNormalization, Activation, Add, UpSampling2D, LeakyReLU, Dense
from keras.optimizers import Adam
from dataProcess import dataProcess
import keras.backend as K

# @author: resumebb
# @time: 2020/11/21

# 基于对抗生成网络实现超分辨率图像重建
class SRGAN():

    #采用VGG19的第九层特征层
    def VGG(self):
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.HR_shape)
        img_features = [vgg.layers[9].output]
        return Model(vgg.input, img_features)

    # 产生网络，用于生成高分辨率图像
    def generatorNet(self):
        def residual_block(input, filter):
            layer = Conv2D(filter, kernel_size=3, strides=1, padding='same')(input)
            # 衰减率暂时设0.8，不考虑性能，优先考虑回环检测需求
            layer = BatchNormalization(momentum=0.8)(layer)
            layer = Activation('relu')(layer)
            layer = Conv2D(filter, kernel_size=3, strides=1, padding='same')(layer)
            layer = BatchNormalization(momentum=0.8)(layer)
            layer = Add()([layer,input])
            return layer

        def deConv(input):
            layer = UpSampling2D(size=2)(input)
            layer = Conv2D(256, kernel_size=3, strides=1, padding='same')(layer)
            layer = Activation('relu')(layer)
            return layer

        #第一部分,传入低分辨率图像
        LR_input = Input(shape=self.LR_shape)
        layer1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(LR_input)
        layer1 = Activation('relu')(layer1)

        #第二部分，经过b个残差结构块
        layer2 = residual_block(layer1, 64)
        for _ in range(self.b_residual_blocks - 1):
            layer2 = residual_block(layer2, 64)

        #第三部分，上采样图像放大为原来的4倍
        layer3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer2)
        layer3 = BatchNormalization(momentum=0.8)(layer3)
        layer3 = Add()([layer3,layer1])

        res1 = deConv(layer3)
        res2 = deConv(res1)
        genernator_HR = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(res2)

        # 返回原低分辨率图像以及生成的伪高清图像
        return Model(LR_input, genernator_HR)

    # 鉴别网络，用于将生成的伪高清图像与真实的高清图像做比较得到loss
    def discriminatorNet(self):

        # 中间的6个卷积块
        def conv_block(input, filter, strides=1, BN=True):
            block = Conv2D(filter, kernel_size=3, strides=strides, padding='same')(input)
            #参数慢慢调吧
            block = LeakyReLU(alpha=0.2)(block)
            if BN:
                block = BatchNormalization(momentum=0.8)(block)
            return block

        # 过滤器和步长参考的论文的结构图上的
        input = Input(shape=self.HR_shape)
        layer1 = conv_block(input=input, filter=64, BN=False)
        layer2 = conv_block(layer1,64,strides=2)
        layer3 = conv_block(layer2, 128)
        layer4 = conv_block(layer3, 128, strides=2)
        layer5 = conv_block(layer4, 256)
        layer6 = conv_block(layer5, 256, strides=2)
        layer7 = conv_block(layer6, 512)
        layer8 = conv_block(layer7, 512,strides=2)
        layer9 = Dense(1024)(layer8)
        layer10 = LeakyReLU(alpha=0.2)(layer9)
        res = Dense(1,activation='sigmoid')(layer10)
        return Model(input, res)

    def learning(self, model, epoch):
        if epoch % 20000 == 0 and epoch != 0:
            for _ in model:
                lr = K.get_value(model.opt.lr)
    def __init__(self):
        #低分辨率初始化
        self.LR_height = 128
        self.LR_width = 128
        self.channels = 3
        self.LR_shape = (self.LR_height, self.LR_width, self.channels)

        #高分辨率初始化
        self.HR_height = self.LR_height * 4
        self.HR_width = self.LR_width * 4
        self.HR_shape = (self.HR_height, self.HR_width, self.channels)

        #b个残差块
        self.b_residual_blocks = 16

        #优化器，初始学习率0.002
        optimizer = Adam(0.0002, 0.9)
        # 数据集
        self.datasets_name = "DIV"
        self.dataProcess = dataProcess(data_name=self.datasets_name, img=(self.HR_height, self.HR_width))

        self.vgg = self.VGG()
        self.vgg.trainable = False

        #建立生成网络
        self.generator = self.generatorNet()
        self.generator.summary()

        # 建立鉴别网络
        patch = int(self.HR_height / 2**4)
        self.discriminator_patch = (patch, patch, 1)
        self.discriminator = self.discriminatorNet()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        self.discriminator.summary()

        LR_img = Input(shape=self.LR_shape)
        gen_HR = self.generator(LR_img)
        gen_HR_features = self.vgg(gen_HR)

        self.discriminator.trainable = False
        validity = self.discriminator(gen_HR)
        self.combined = Model(LR_img, [validity, gen_HR_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[5e-1,1],optimizer=optimizer)

    def learining(self, model, epoch):
        if epoch % 20000 == 0 and epoch != 0:
            for _ in model:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
                print("learning rate ->{}",format(lr * 0.5))

    def train(self, epochs, init_epoch=0, batch_size=1, example=50):
        begin_time = datetime.datetime.now()
        if init_epoch != 0:
            self.generator.load_weights("weights/%s/gen_epoch%d.h5" % (self.datasets_name, init_epoch), skip_mismatch=True)
            self.discriminator.load_weights("weights/%s/dis_epoch%d.h5" % (self.dataset_name, init_epoch), skip_mismatch=True)

        for epoch in range(init_epoch, epochs):
            self.learining([self.combined, self.discriminator], epoch)
            HR_img, LR_img = self.dataProcess.process(batch_size)
            gen_HR = self.generator.predict(LR_img)
            valid = np.ones((batch_size,) + self.discriminator_patch)
            gen = np.zeros((batch_size,) + self.discriminator_patch)
            origin_loss = self.discriminator.train_on_batch(HR_img, valid)
            gen_loss = self.discriminator.train_on_batch(gen_HR, gen)
            d_loss = 0.5 * np.add(origin_loss, gen_loss)

            HR_img, LR_img = self.dataProcess.process(batch_size)
            valid = np.ones((batch_size,) + self.discriminator_patch)
            img_features = self.vgg.predict(HR_img)
            g_loss = self.combined.train_on_batch(LR_img, [valid, img_features])
            print(d_loss, g_loss)
            end_time = datetime.datetime.now()
            time = end_time - begin_time
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] time: %s " \
                  % (epoch,
                     epochs,
                     d_loss[0],
                     100 * d_loss[1],
                     g_loss[1],
                     g_loss[2],
                     time))

            if epoch % example == 0:
                self.restore(epoch)
                if epoch % 500 == 0 and epoch != init_epoch:
                    # 500代保存一次
                    os.makedirs('weights/%s' % self.datasets_name, exist_ok=True)
                    self.generator.save_weights("weights/%s/gen_epoch%d.h5" % (self.datasets_name, epoch))
                    self.dicriminator.save_weights("weights/%s/dis_epoch%d.h5" % (self.datasets_name, epoch))


    def restore(self, epoch):
        os.makedirs('images/%s' % self.datasets_name, exist_ok=True)
        HR_img, LR_img = self.dataProcess.process(batch_size=2, data_type=True)
        gen_HR= self.generator.predict(LR_img)

        HR_img = 0.5 * HR_img + 0.5
        LR_img = 0.5 * LR_img + 0.5
        gen_HR = 0.5 * gen_HR + 0.5

        #将图像进行拼接方便对比
        title = ['Generated', 'Original']

        fig, axs = plt.subplots(2, 2)
        count = 0
        for row in range(1):
            for col, img in enumerate([gen_HR, HR_img]):
                axs[row, col].imshow(img[row])
                axs[row, col].set_title(title[col])
                axs[row, col].axis('off')
            count += 1
        fig.savefig("images/%s/%d.png" % (self.datasets_name, epoch))
        plt.close()
        for _ in range(1):
            fig = plt.figure()
            plt.imshow(LR_img[_])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.datasets_name, epoch, _))
            plt.close()


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=3000,init_epoch = 0, batch_size=2, example=5)

