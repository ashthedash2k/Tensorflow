import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#loading directories
train_path = '/Users/ashleyczumak/Deeplearning/chest_xray/train'
test_path = '/Users/ashleyczumak/Deeplearning/chest_xray/test'
valid_path = '/Users/ashleyczumak/Deeplearning/chest_xray/val'


BATCH_SIZE = 10

train_batches = ImageDataGenerator(
     rescale=1/255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True
).flow_from_directory(
    directory=train_path,
    target_size=(150, 150),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_batches = ImageDataGenerator(
    rescale=1/255.
).flow_from_directory(
    directory=test_path,
    target_size=(150, 150),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

valid_batches = ImageDataGenerator( rescale=1/255.
).flow_from_directory(
    directory=valid_path,
    target_size=(150, 150),
    classes=['NORMAL', 'PNEUMONIA'],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(32, (3, 3),activation=('relu'),input_shape=(150,150,3)))
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
model.fit(train_batches, epochs=20)


print(f'final{model.evaluate(test_batches)}')
