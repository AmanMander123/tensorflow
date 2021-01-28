import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import matplotlib.pyplot as plt

path_cats_and_dogs = f"{getcwd()}/../tmp/cats-and-dogs.zip"


local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('../tmp')
zip_ref.close()


#print(len(os.listdir('../tmp/PetImages/Cat')))
#print(len(os.listdir('../tmp/PetImages/Dog')))


os.mkdir('../tmp/cats-v-dogs/')
os.mkdir('../tmp/cats-v-dogs/training/')
os.mkdir('../tmp/cats-v-dogs/testing/')
os.mkdir('../tmp/cats-v-dogs/training/cats/')
os.mkdir('../tmp/cats-v-dogs/training/dogs/')
os.mkdir('../tmp/cats-v-dogs/testing/cats/')
os.mkdir('../tmp/cats-v-dogs/testing/dogs/')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_images = os.listdir(SOURCE)
    training = random.sample(all_images, round(len(all_images)*SPLIT_SIZE))
    validation = [x for x in all_images if x not in training]
    for item in training:
        full_path = os.path.join(SOURCE,item)
        if os.path.getsize(full_path) != 0:
            copyfile(full_path, os.path.join(TRAINING, item))
    for item in validation:
        full_path = os.path.join(SOURCE,item)
        if os.path.getsize(full_path) != 0:
            copyfile(full_path, os.path.join(TESTING, item))

CAT_SOURCE_DIR = '../tmp/PetImages/Cat/'
TRAINING_CATS_DIR = '../tmp/cats-v-dogs/training/cats/'
TESTING_CATS_DIR = '../tmp/cats-v-dogs/testing/cats/'
DOG_SOURCE_DIR = '../tmp/PetImages/Dog/'
TRAINING_DOGS_DIR = '../tmp/cats-v-dogs/training/dogs/'
TESTING_DOGS_DIR = '../tmp/cats-v-dogs/testing/dogs/'

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('../tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('../tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('../tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('../tmp/cats-v-dogs/testing/dogs/')))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

TRAINING_DIR = '../tmp/cats-v-dogs/training/'
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagenerator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=10,
    class_mode='binary',
    target_size=(150, 150))

VALIDATION_DIR = '../tmp/cats-v-dogs/testing/'
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagenerator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=10,
    class_mode='binary',
    target_size=(150, 150))

history = model.fit(train_datagenerator,
                    epochs=2,
                    verbose=1,
                    validation_data=validation_datagenerator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and Validation Loss')
plt.show()
