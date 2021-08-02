import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
import numpy as np
# 构建原始全连接网络模型并进行训练，最后保存该网络模型


# mnist数据导入
(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()
train_imgs, test_imgs = (train_imgs / 255, test_imgs / 255)
# 模型构建
train_imgs = np.expand_dims(train_imgs, axis=-1)
test_imgs = np.expand_dims(test_imgs, axis=-1)

model = tf.keras.Sequential()
model.add(Conv2D(64, 3, input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
# 模型编译
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 模型训练
history = model.fit(x=train_imgs,
                    y=train_labels,
                    batch_size=64,
                    epochs=10,
                    validation_data=(test_imgs, test_labels),
                    verbose=1)
# 模型保存
model.save('./model.h5')
