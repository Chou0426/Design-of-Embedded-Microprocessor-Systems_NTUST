import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from time import sleep
from serial import Serial

#載入dataset
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()

train_img = train_img / 255.0
test_img = test_img / 255.0

# plt.imshow(np.reshape(train_img[100, :], (28, 28)),cmap = plt.get_cmap('gray'))
# plt.show()
# print(train_label[100])

#查看train, test資料筆數
print('train_img_num = {}'.format(train_img.shape))
print('test_img_num = {}'.format(test_img.shape))

#進行 CNN 模型的建構
# model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = 'relu'),
#         tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2, padding = 'SAME'),
#         tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = 'relu'),
#         tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2, padding = 'SAME'),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1024, tf.nn.relu),

#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(10, tf.nn.softmax),
# ])

# model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

# train_img = np.reshape(train_img, [-1, 28, 28, 1])

# model.fit(train_img, train_label, validation_split = 0.2)

# # 儲存model
# model.save('./mnist_model')

# model.summary()



# 載入先前儲存的model
# model = tf.keras.models.load_model('./mnist_model')
# # Reshape test image [張數,長,寬] -> [張數,長,寬,維度(gray = 1)]
# x = np.reshape(test_img[30, :], [-1, 28, 28, 1])
# # 預測結果
# y = np.argmax(model.predict(x))
# print("Predict:", y)
# plt.imshow(np.reshape(test_img[30, :], (28,28)), cmap = plt.get_cmap('gray'))
# plt.show()

'''arduino test'''
ser = Serial('COM3', 9600, timeout = 1)
# ser.write(bytes([y]))
# time.sleep(3)

model = tf.keras.models.load_model('./mnist_model')
x = []
x.append(test_img[3])
x.append(test_img[2])
x.append(test_img[1])
x.append(test_img[30])
x.append(test_img[4])
x.append(test_img[15])
x.append(test_img[11])
x.append(test_img[0])
x.append(test_img[61])
x.append(test_img[7])

for i in range(10):
    x[i] = np.reshape(x[i], [-1, 28, 28, 1])
    y = np.argmax(model.predict(x[i]))
    print("Predict:", y)
    plt.imshow(np.reshape(x[i], (28,28)), cmap = plt.get_cmap('gray'))
    plt.show()
    ser.write(bytes([y]))
    time.sleep(3)