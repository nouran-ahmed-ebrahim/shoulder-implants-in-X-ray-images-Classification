import cv2
import numpy as np
import os
from random import shuffle

import skimage
import tensorflow as tf
from keras.layers import Dropout, Conv2D, Flatten, Dense, BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from numpy import expand_dims
from tflearn import softmax
from tqdm import tqdm

Train_Data_Directory = r'train'
Test_Data_Directory = r'test'
Image_Size = 224


def create_label(image_name):
    Factory_Label = image_name.split('.')[0]
    if Factory_Label == "Cofield":
        return np.array([1, 0, 0, 0])
    elif Factory_Label == 'Depuy':
        return np.array([0, 1, 0, 0])
    elif Factory_Label == 'Tornier':
        return np.array([0, 0, 1, 0])
    elif Factory_Label == 'Zimmer':
        return np.array([0, 0, 0, 1])


def create_train_data():
    training_data = []
    for image in tqdm(os.listdir(Train_Data_Directory)):
        image_path = os.path.join(Train_Data_Directory, image)
        img_data = cv2.imread(image_path, 1)
        img_data = cv2.resize(img_data, (222, 222))
        img_data = add_border(img_data)
        img_data = hisogram_Normalization(img_data)
        training_data.append([np.array(img_data), create_label(image)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for image in tqdm(os.listdir(Test_Data_Directory)):
        image_path = os.path.join(Test_Data_Directory, image)
        img_data = cv2.imread(image_path, 1)
        img_data = cv2.resize(img_data, (222, 222))
        img_data = add_border(img_data)
        img_data = hisogram_Normalization(img_data)
        testing_data.append([np.array(img_data), create_label(image)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def add_border(old_im):
    return cv2.copyMakeBorder(old_im, 1, 1, 1, 1, cv2.BORDER_CONSTANT)


def hisogram_Normalization(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
    return equalized_img


def creat_model():
    mode_input = tf.keras.Input(shape=(224, 224, 3))
    x = Conv2D(32, kernel_size=3, activation='relu')(mode_input)
    x = Conv2D(32, kernel_size=3, activation='relu')(mode_input)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output1 = tf.keras.layers.Dense(4, activation='softmax', name="label")(x)
    model = tf.keras.Model(inputs=mode_input, outputs=[output1])
    model.compile(loss={"label": tf.keras.losses.CategoricalCrossentropy(from_logits=False)},
                  optimizer='adam', metrics=['accuracy'])
    history_object = model.fit(X_train, y_train, validation_split=0.17, epochs=10, shuffle=True)
    print(history_object)
    print(model.evaluate(X_test, y_test))
    model.save('my_model.h5')
    return model


if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy', allow_pickle=True)

else:
    train_data = create_train_data()

if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()

X_train = np.array([i[0] for i in train_data])
y_train = np.array([i[1] for i in train_data])

X_test = np.array([i[0] for i in test_data])
y_test = np.array([i[1] for i in test_data])

if os.path.exists('my_model.h5'):
    model = load_model('my_model.h5')
else:
    model = creat_model()

img = load_img('Cofield.45.jpg', target_size=(224, 224))
img_array = img_to_array(img)
img_array = expand_dims(img_array, 0)  # Create a batch
predictions = model.predict(img_array)
print(f"Cofield: {predictions[0][0] * 100}, Depuy: {predictions[0][1] * 100}, Tornier: {predictions[0][2]* 100}, Zimmer: {predictions[0][3]*100}")
