import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#loading directories
train_path = '/Users/ashley/Deeplearning/chest_xray/train'
test_path = '/Users/ashley/Deeplearning/chest_xray/test'
valid_path = '/Users/ashley/Deeplearning/chest_xray/val'

BATCH_SIZE = 10

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
).flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
).flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
).flow_from_directory(
    directory=valid_path,
    target_size=(224, 224),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

#VISUALIZATION
images, labels = next(train_batches)
def plot(img):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for imgs, ax, in zip(img, axes):
        ax.imshow(imgs)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plot(images)
print(labels)

#MODEL 
model = Sequential()
model.add(Conv2D(32, (3, 3),activation=('relu'),input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation=('relu'))) #extra
model.add(MaxPooling2D(pool_size=(2, 2))) #extra

model.add(Conv2D(64, (3, 3),activation=('relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64,activation=('relu')))
model.add(Dropout(0.2))
model.add(Dense(1,activation=('sigmoid')))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(test_batches, epochs=17)
