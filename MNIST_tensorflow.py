import tensorflow as tf

data = tf.keras.datasets.mnist

(training_images, training_labels) , (test_images, test_labels) = data.load_data()


training_images = training_images /255.0
test_images = test_images/255.0

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

net.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
EPOCHS = 4
net.fit(training_images, training_labels, epochs=EPOCHS)
