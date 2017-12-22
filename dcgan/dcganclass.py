# -*- encoding: utf-8 -*-
'''
Created on 2017年12月19日

@author: Guo
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Activation,BatchNormalization,\
Dense,Reshape,UpSampling2D,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam,SGD
from read_data import read_data
import _thread
import math

class dcgan:
    epoch = 0
    height = 0
    width = 0
    batch_size = 0
    
    def __init__(self, Data = None, Epoch = 100, img_height = 28, img_width = 28, Batch_size = 128):
        if Data.all() == 0 :
            print("No input data!!!")
        self.gm = self.generator_model()
        self.dm = self.discriminator_model()
        self.gd = self.generator_containing_discriminator(self.gm,self.dm)
        self.epoch = Epoch
        self.height = img_height
        self.width = img_width
        self.batch_size = Batch_size
        self.data = Data.astype(np.float32)
        
    def generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        print("generator_model:")
        model.summary()
        return model

    def discriminator_model(self):
        model = Sequential()
        model.add(Convolution2D(
                            64, 5, 5,
                            border_mode='same',
                            input_shape=(28, 28, 1)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print("discriminator_model:")
        model.summary()
        return model

    def generator_containing_discriminator(self,generator, discriminator):
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False #D不可训练
        model.add(discriminator)
        return model
    
    def log(self,d_loss,g_loss,accuracy):       
        loss_log = os.path.join("log","loss.txt")       
        f = open(loss_log,"a")
        line = ("%f    %f    %f\n" %(d_loss,g_loss,accuracy))
        f.write(line) 
        f.close
    
    def generator(self):
        self.gm.compile(loss='binary_crossentropy', optimizer="SGD")
        self.gm.load_weights('generator')
        noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
        generated_img = self.gm.predict(noise,batch_size=self.batch_size,verbose=1)
        filename = "generated_img.png" 
        savapath = os.path.join("samples",filename)
        for i in range(16):
            plt.subplot(4, 4, i+1)
            image = generated_img[i, :, :, :]
            image = np.reshape(image, [self.height, self.height])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(savapath)
            plt.close('all')
            print("saved")
        
        
    def train(self):
        X_train = self.data
        #g_optim = Adam(self, lr=0.01, beta_1=0.9, beta_2=0.999)
        #d_optim = Adam(self, lr=0.01, beta_1=0.9, beta_2=0.999)
        g_optim = SGD(lr=0.005, momentum=0.9, nesterov=True)
        d_optim = SGD(lr=0.005, momentum=0.9, nesterov=True)
        self.gm.compile(loss='binary_crossentropy', optimizer="SGD")
        self.gd.compile(loss='binary_crossentropy', optimizer=g_optim) #in fact gm.train = gd.train (trainable)
        self.dm.trainable = True
        self.dm.compile(loss='binary_crossentropy', optimizer=d_optim)
        print("Data_Size: ",self.data.shape)
        print("Image_Size: ",(self.height,self.width))
        print("Batch_Size: ",self.batch_size)
        print("Total_Epoch: ",self.epoch)
        print("_________________________________________________________________")
        print("Start Train! \n")
        for epoch in range(self.epoch):    
            print("Current epoch ", epoch)
            print("Number of batches", int(X_train.shape[0]/self.batch_size))
            for index in range(int(X_train.shape[0]/self.batch_size)):
                noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
                image_batch = X_train[index*self.batch_size:(index+1)*self.batch_size]
                #image_batch = image_batch.reshape(-1,self.height,self.height,1);
                generated_images = self.gm.predict(noise, verbose=0)
                #the same size as real images(batch_size * height * width)
                if index % 30 == 0:
                    filename = "mnist_%d_%d.png" %(epoch,index)
                    savapath = os.path.join("samples",filename)
                    for i in range(16):
                        plt.subplot(4, 4, i+1)
                        image = generated_images[i, :, :, :]
                        image = np.reshape(image, [self.height, self.height])
                        plt.imshow(image, cmap='gray')
                        plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(savapath)
                    plt.close('all')
                    print("saved")
                X = np.concatenate((image_batch, generated_images)) #fuse real images and fake images
                y = [1] * self.batch_size + [0] * self.batch_size #with their real/fake labels
                #print(abs(y-self.dm.predict(X)))
                
                accuracy = sum(abs(np.asarray(y)-np.round(self.dm.predict(X)).reshape(-1))) / (2*self.batch_size)
                d_loss = self.dm.train_on_batch(X, y) #train discriminator_model
                noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100]) #why a new noise
                self.dm.trainable = False
                g_loss = self.gd.train_on_batch(noise, [1] * self.batch_size) #train generator_containing_discriminator
                #g_loss:why noise in but loss on true? generator wants that discriminator to be unable to discriminate
                #in another word,the goal of generator is to make discriminator doing a wrong judgement
                self.dm.trainable = True 
                print("epcho:%d | batch:%d | d_loss : %f | g_loss : %f | accuracy : %f" % (epoch,index, d_loss,g_loss,accuracy))
                _thread.start_new_thread ( self.log, (d_loss, g_loss,accuracy ))
            #epoch training...  
            self.gm.save_weights('generator', True) #sava model index every epoch
            self.dm.save_weights('discriminator', True)

if __name__ =="__main__":
    loss_log = os.path.join("log","loss.txt")
    try:
        os.remove(loss_log) 
    except:
        pass
    h = 28
    w = 28
    data_mnist,_,_,_ = read_data()
    face_dcgan = dcgan(Epoch = 100,img_height = h,img_width = w,Data = data_mnist)
    face_dcgan.train()