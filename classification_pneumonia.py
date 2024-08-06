import numpy as np 
import pandas as pd 
import time
import itertools

import os
import shutil

from sklearn.metrics import confusion_matrix, classification_report, f1_score 
import splitfolders

from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121

# Constants
FOLDERS = ['train', 'test', 'val']
DIR_INPUT = r'C:\Users\sivak\Downloads\pneumonia\chest_xray'
DIR_WORKING = './'
DIR_MODELS = os.path.join(DIR_WORKING, 'models')
DIR_TRAIN = os.path.join(DIR_WORKING, 'train')
DIR_VAL = os.path.join(DIR_WORKING, 'val')
DIR_TEST = os.path.join(DIR_WORKING, 'test')
CLASS_LIST = ['normal', 'bacteria', 'virus']

# Set seeds for reproducibility 
SEED = 1985
tf.random.set_seed(SEED)
np.random.seed(SEED)

def images_by_class(path, folders):
    """
    Loop through the path/folders count the number and proportions of normal, bacteria and viral xrays
    """
    normal, bacterial, viral = 0, 0, 0
    msg = '{:8} {:8} {:11} {:7} {:9} {:11} {:7}'.format('folder', 'normal', 'bacterial', 'viral',
                                                        'normal %', 'bacterial %', 'viral %')
    print(msg)  
    print("-" * len(msg))
    
    for folder in folders:
        for dirname, _, filenames in os.walk(os.path.join(path, folder)):
            for filename in filenames:
                if 'normal' in dirname.lower():
                    normal += 1
                if 'bacteria' in filename.lower():
                    bacterial += 1
                if 'virus' in filename.lower():
                    viral += 1  
                    
        total = normal + bacterial + viral   
        if total > 0:
            n = round(normal / total, 2) * 100
            b = round(bacterial / total, 2) * 100
            v = round(viral / total, 2) * 100
        else:
            n, b, v = 0, 0, 0
        
        print("{:6} {:8} {:11} {:7} {:10} {:12} {:7} ".format(folder, normal, bacterial, viral, n, b, v))
        normal, bacterial, viral = 0, 0, 0

# Images by class in the input directory
images_by_class(DIR_INPUT, FOLDERS)

def create_dir(dir_path, folder, verbose=True):
    """
    Create the dir_path/folder if it doesn't already exist
    """
    msg = ""
    folder_path = os.path.join(dir_path, folder)
    
    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
            msg = folder_path + ' created'
        except OSError as err:
            print('Error creating folder:{} with error:{}'.format(folder_path, err))
    if verbose:
        print(msg)
        
    return folder_path

def split_files_by_class(working_dir, input_dir, folders):
    """
    Move images from the read-only input dir to the working dir
    Split the images by normal, bacteria and viral images
    """
    msg = "Separating Images By Class"
    print(msg)
    print("-" * len(msg))
    for folder in folders:
        class_folder = os.path.join(working_dir, folder)
        if os.path.exists(class_folder):
            shutil.rmtree(class_folder)
        create_dir(working_dir, folder)

        for dirname, _, filenames in os.walk(os.path.join(input_dir, folder)):
            if 'normal' in dirname.lower():
                copyto = os.path.join(working_dir, folder, 'normal')
                try:
                    shutil.copytree(dirname, copyto)
                except OSError as err:
                    print('Error copying normal directory from {} to {}:{}'.format(dirname, copyto, err))
            else:
                create_dir(working_dir, os.path.join(folder, "bacteria"))
                create_dir(working_dir, os.path.join(folder, "virus"))

                for filename in filenames:
                    if 'bacteria' in filename.lower(): 
                        new_dir = os.path.join(working_dir, folder, 'bacteria')
                        try:
                            shutil.copy2(os.path.join(dirname, filename), new_dir)
                        except OSError as err:
                            print("Error copying file {} to {} with error:{}".format(filename, new_dir, err))
                    if 'virus' in filename.lower():
                        new_dir = os.path.join(working_dir, folder, 'virus')
                        try:
                            shutil.copy2(os.path.join(dirname, filename), new_dir)
                        except OSError as err:
                            print("Error copying file {} to {} with error:{}".format(filename, new_dir, err))

    print("\n Images By Class After Separating Classes")
    print("-" * 67)
    images_by_class(working_dir, folders)

create_dir(DIR_WORKING, 'models', verbose=False)
split_files_by_class(DIR_WORKING, DIR_INPUT, FOLDERS)

def move_val_to_train(working_dir, class_names):
    for class_name in class_names:
        dir_from = os.path.join(working_dir, "val", class_name)
        dir_to = os.path.join(working_dir, "train", class_name)
        for dirname, _, filenames in os.walk(dir_from):
            for filename in filenames:
                try:
                    shutil.copy2(os.path.join(dirname, filename), dir_to)
                except OSError as err:
                    print("Error moving file:{} with Error:{}".format(filename, err))
    
    temp_dir = os.path.join(working_dir, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    os.renames(os.path.join(working_dir, 'train'), temp_dir)
    shutil.rmtree(os.path.join(working_dir, 'val'))
    
    print("\n Images By Class After Moving Val to Train")
    print("-" * 67)
    images_by_class(working_dir, ["test", 'temp'])

move_val_to_train(DIR_WORKING, CLASS_LIST)


def resample_train_val_images(working_dir, seed=SEED, split=(0.80, 0.20)):
    input_dir  = os.path.join(working_dir, 'temp')
    output_dir = os.path.join(working_dir)
    splitfolders.ratio(input_dir, output_dir, seed=seed, ratio=split)
    try:
        shutil.rmtree(input_dir)
    except OSError as err:
        print("Error removing directory:{} with error:{}".format(input_dir, err))
    print("\n Images By Class After Resampling")
    print("-" * 67)
    images_by_class(working_dir, ["train", "test", "val"])

resample_train_val_images(DIR_WORKING)

BATCH_SIZE = 32
IMG_SIZE = [150, 150]

train_images = ImageDataGenerator(rescale=1./255)
val_images = ImageDataGenerator(rescale=1./255)
test_images = ImageDataGenerator(rescale=1./255)

train_gen = train_images.flow_from_directory(
    DIR_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    classes=CLASS_LIST,
    seed=SEED
)

val_gen = val_images.flow_from_directory(
    DIR_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    classes=CLASS_LIST,
    seed=SEED
)

test_gen = test_images.flow_from_directory(
    DIR_TEST,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    classes=CLASS_LIST,
    seed=SEED,
    shuffle=False
)

steps_per_epoch = train_gen.n // train_gen.batch_size
validation_steps = val_gen.n // val_gen.batch_size
print("steps per epoch:{} \nvalidation steps:{}".format(steps_per_epoch, validation_steps))

def show_images(generator, y_pred=None):
    labels = dict(zip([0, 1, 2], CLASS_LIST))
    x, y = generator.next()
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[i])]))
    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: 'normal', 'bacteria', 'virus'
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=10,  # Adjust epochs as needed
    validation_data=val_gen,
    validation_steps=validation_steps
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate predictions
test_gen.reset()
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# Print classification report
print(classification_report(true_classes, y_pred_classes, target_names=class_labels))

# Calculate and print F1 score
f1 = f1_score(true_classes, y_pred_classes, average='weighted')
print(f'Weighted F1 Score: {f1:.4f}')
