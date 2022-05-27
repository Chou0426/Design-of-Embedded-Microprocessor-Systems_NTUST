import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
from keras import utils


# 載入先前儲存的model
model = tf.keras.models.load_model('./mnist_model')
# Reshape test image [張數,長,寬] -> [張數,長,寬,維度(gray = 1)]
x = np.reshape(test_img[18, :], [-1, 28, 28, 1])
# 預測結果
y = np.argmax(model.predict(x))
print("Predict:", y)
plt.imshow(np.reshape(test_img[18, :], (28,28)), cmap = plt.get_map('gray'))
plt.show()