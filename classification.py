import cv2
import numpy as np
import os
from random import shuffle

from skimage.exposure import equalize_hist
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
import skimage
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class Classification:
    Train_Data_Directory = r'train'
    Test_Data_Directory = r'test'
    Image_Size = 224
    LR = 0.001
    MODEL_NAME = 'shoulderClassification'

    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.prepare_data()
        self.model_arch()
        self.get_model()

    def create_label(self, image_name):
        Factory_Label = image_name.split('.')[0]
        if Factory_Label == "Cofield":
            return np.array([1, 0, 0, 0])
        elif Factory_Label == 'Depuy':
            return np.array([0, 1, 0, 0])
        elif Factory_Label == 'Tornier':
            return np.array([0, 0, 1, 0])
        elif Factory_Label == 'Zimmer':
            return np.array([0, 0, 0, 1])

    def create_train_data(self):
        training_data = []
        for image in os.listdir(self.Train_Data_Directory):
            image_path = os.path.join(self.Train_Data_Directory, image)
            img_data = cv2.imread(image_path, 0)
            img_data = np.asarray(equalize_hist(img_data) , dtype='uint8')
            img_data = cv2.resize(img_data, (self.Image_Size, self.Image_Size))
            training_data.append([np.array(img_data), self.create_label(image)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def create_test_data(self):
        testing_data = []
        for image in os.listdir(self.Test_Data_Directory):
            image_path = os.path.join(self.Test_Data_Directory, image)
            img_data = cv2.imread(image_path, 0)
            img_data = np.asarray(equalize_hist(img_data) , dtype='uint8')
            img_data = cv2.resize(img_data, (self.Image_Size, self.Image_Size))
            testing_data.append([np.array(img_data), self.create_label(image)])
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data

    def prepare_data(self):
        if os.path.exists('train_data.npy'):
            train_data = np.load('train_data.npy', allow_pickle=True)

        else:
            train_data = self.create_train_data()

        if os.path.exists('test_data.npy'):
            test_data = np.load('test_data.npy', allow_pickle=True)
        else:
            test_data = self.create_test_data()

        self.X_train = np.array([i[0] for i in train_data]).reshape(-1, self.Image_Size, self.Image_Size, 1)
        self.y_train = [i[1] for i in train_data]

        self.X_test = np.array([i[0] for i in test_data]).reshape(-1, self.Image_Size, self.Image_Size, 1)
        self.y_test = [i[1] for i in test_data]

    def model_arch(self):
        conv_input = input_data(shape=[None, self.Image_Size, self.Image_Size, 1], name='input')
        conv1 = conv_2d(conv_input, 32, 3, activation='relu')
        pool1 = max_pool_2d(conv1, 2)

        conv2 = conv_2d(pool1, 64, 3, activation='relu')
        pool2 = max_pool_2d(conv2, 2)

        conv3 = conv_2d(pool2, 128, 3, activation='relu')
        pool3 = max_pool_2d(conv3, 2)

        fully_layer = fully_connected(pool3, 256, activation='relu')
        fully_layer = dropout(fully_layer, 0.5)

        cnn_layers = fully_connected(fully_layer, 4, activation='softmax')

        cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy',
                                name='targets')
        self.model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
        # print(X_train.shape)

    def get_model(self):
        if os.path.exists('model.tfl.meta'):
            self.model.load('./model.tfl')
        else:
            self.model.fit({'input': self.X_train}, {'targets': self.y_train}, n_epoch=20,
                      validation_set=({'input': self.X_test}, {'targets': self.y_test}),
                      snapshot_step=500, show_metric=True, run_id= self.MODEL_NAME)
            self.model.save('model.tfl')



    def print_c(self, img):
        test_img = cv2.resize(img, (self.Image_Size, self.Image_Size))
        test_img = cv2.equalizeHist(test_img)
        test_img = test_img.reshape(self.Image_Size, self.Image_Size, 1)
        prediction = self.model.predict([test_img])[0]
        skimage.io.imshow(img)

        print(f"Cofield: {prediction[0]}, Depuy: {prediction[1]}, Tornier: {prediction[2]}, Zimmer: {prediction[3]}")
        plt.show()


def _main_():
    c = Classification()
    path1 = cv2.imread('Cofield.45.jpg', 0)

    c.print_c(path1)

_main_()
'''
def _main_():
    path1 = cv2.imread('Cofield.45.jpg', 0)

    print_c(path1)

_main_()

test_img = cv2.resize(path, (self.Image_Size, self.Image_Size))
    test_img = cv2.equalizeHist(test_img)
    test_img = test_img.reshape(self.Image_Size, self.Image_Size, 1)
    prediction = self.model.predict([test_img])[0]
    skimage.io.imshow(path)
    
    
    
    //
    
     if prediction[0] == 1:
        print("Cofield")
    elif prediction[1] == 1:
        print("Depuy")
    elif prediction[2] == 1:
        print("Tornier")
    elif prediction[3] == 1:
        print("Zimmer")
'''