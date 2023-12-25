## Importing Libraries ## 
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import os
import random
import pathlib
import warnings
import itertools
import math
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

K.clear_session()

# Replace the following two lines with your dataset directories
train_folder = "path_to_your_train_dataset_directory"
test_folder = "path_to_your_test_dataset_directory"

def count_files(rootdir):
    '''counts the number of files in each subfolder in a directory'''
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            print("There are " + str(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])) + " files in " + str(path.name))

########### Pre-processing #########
image_folder = "malignant"
number_of_images = 2

def Preprocess():
    j = 1
    for i in range(number_of_images):
        folder = os.path.join(test_folder, image_folder)
        a = random.choice(os.listdir(folder))
        image = Image.open(os.path.join(folder, a))
        image_duplicate = image.copy()
        plt.figure(figsize=(10, 10))
        plt.subplot(number_of_images, 2, j)
        plt.title(label='Original', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image)
        j += 1
        image1 = ImageEnhance.Color(image_duplicate).enhance(1.35)
        image1 = ImageEnhance.Contrast(image1).enhance(1.45)
        image1 = ImageEnhance.Sharpness(image1).enhance(2.5)
        plt.subplot(number_of_images, 2, j)
        plt.title(label='Processed', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image1)
        j += 1

Preprocess()

select_folder = "malignant"
rows, columns = 1, 5
display_folder = os.path.join(train_folder, select_folder)
total_images = rows * columns
fig = plt.figure(1, figsize=(30, 10))
for i, j in enumerate(os.listdir(display_folder)):
    img = plt.imread(os.path.join(train_folder, select_folder, j))
    fig = plt.subplot(rows, columns, i + 1)
    fig.set_title(select_folder, pad=11, size=20)
    plt.imshow(img)
    if i == total_images - 1:
        break

images = []
for image_folder in sorted(os.listdir(train_folder)):
    leaf = os.listdir(os.path.join(train_folder, image_folder))
    img_selected = np.random.choice(leaf)
    images.append(os.path.join(train_folder, image_folder, img_selected))

fig = plt.figure(1, figsize=(30, 10))
for subplot, image_ in enumerate(images):
    category = image_.split('/')[-2]
    imgs = plt.imread(image_)
    a, b, c = imgs.shape
    fig = plt.subplot(7, 5, subplot + 1)
    fig.set_title(category, pad=10, size=18)
    plt.imshow(imgs)

plt.tight_layout()

n_cat = 2
batch_size = 48
batch_size_predict = 128
input_shape = (299, 299)

## Model Building ##
x_model = Xception(input_shape=list(input_shape) + [3], weights='imagenet', include_top=False)

for layer in x_model.layers:
    layer.trainable = True

for layer in x_model.layers[:85]:
    layer.trainable = False

x_model.summary()

gm_exp = tf.Variable(3., dtype=tf.float32)

def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), axis=[1, 2], keepdims=False) + 1.e-8)**(1./gm_exp)
    return pool

X_feat = Input(x_model.output_shape[1:])
lambda_layer = Lambda(generalized_mean_pool_2d)
lambda_layer.trainable_weights.extend([gm_exp])
X = lambda_layer(X_feat)
X = Dropout(0.05)(X)
X = Activation('relu')(X)
X = Dense(n_cat, activation='softmax')(X)

top_model = Model(inputs=X_feat, outputs=X)
top_model.summary()

X_image = Input(list(input_shape) + [3])
X_f = x_model(X_image)
X_f = top_model(X_f)
model = Model(inputs=X_image, outputs=X_f)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_folder,
    target_size=(299, 299),
    batch_size=48,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_folder,
    target_size=(299, 299),
    batch_size=48,
    class_mode='categorical'
)

class_map = training_set.class_indices
class_map

Model = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

## Accuracy Score Plot ##
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Accuracy_v1_InceptionV3')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label="validation loss")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Loss_v1_InceptionV3')
    plt.show()

plot_accuracy(Model)
plot_loss(Model)

# Using the test dataset
score = model.evaluate_generator(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Replace the following line with your validation dataset directory
validation_data_dir = "path_to_your_validation_dataset_directory"

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical'
)

scores = model.evaluate_generator(validation_generator)
print("Test Accuracy: {:.3f}".format(scores[1]))

category = {0: 'benign', 1: 'malignant'}

def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    plt.title("Prediction - {}".format(category[index]))
    plt.imshow(img_array)

def predict_dir(filedir, model):
    cols = 3
    pos = 0
    images = []

    total_images = len(os.listdir(filedir))
    rows = total_images // cols + 1

    true = filedir.split('/')[-1]

    for i in sorted(os.listdir(filedir)):
        images.append(os.path.join(filedir, i))

    for subplot, imggg in enumerate(images):
        img_ = image.load_img(imggg, target_size=(299, 299))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.

        prediction = model.predict(img_processed)
        index = np.argmax(prediction)
        pred = category.get(index)

        if pred == true:
            pos += 1

    acc = pos / total_images
    print("Accuracy for {}: {:.2f} ({}/{})".format(true, acc, pos=pos, total=total_images))

predict_image(os.path.join(validation_data_dir, 'malignant/1027.jpg'), model)

## Confusion Matrix ##
def labels_confusion_matrix(test_folder):
    folder_path = test_folder
    mapping = {}
    
    for i, j in enumerate(sorted(os.listdir(folder_path))):
        mapping[j] = i

    files = []
    real = []
    predicted = []

    for i in os.listdir(folder_path):
        true = os.path.join(folder_path, i)
        true = true.split('/')[-1]
        true = mapping[true]

        for j in os.listdir(os.path.join(folder_path, i)):
            img_ = image.load_img(os.path.join(folder_path, i, j), target_size=(299, 299))
            img_array = image.img_to_array(img_)
            img_processed = np.expand_dims(img_array, axis=0)
            img_processed /= 255.

            prediction = model.predict(img_processed)
            index = np.argmax(prediction)

            predicted.append(index)
            real.append(true)

    return (real, predicted)

def print_confusion_matrix(real, predicted):
    total_output_labels = 2
    cmap = "OrRd"
    cm_plot_labels = [i for i in range(2)]
    cm = confusion_matrix(y_true=real, y_pred=predicted)
    df_cm = pd.DataFrame(cm, cm_plot_labels, cm_plot_labels)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(15, 10))
    s = sns.heatmap(df_cm, fmt="d", annot=True, cmap=cmap)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

y_true, y_pred = labels_confusion_matrix(test_folder)
print_confusion_matrix(y_true, y_pred)
