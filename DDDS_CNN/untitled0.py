import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import (
    Dropout,
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
    BatchNormalization,
)
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix

def generator(
    dir,
    gen=image.ImageDataGenerator(rescale=1.0 / 255),
    shuffle=True,
    batch_size=1,
    target_size=(24, 24),
    class_mode="categorical",
):

    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode="grayscale",
        class_mode=class_mode,
        target_size=target_size,
    )

# Tải lại mô hình
cnn_model = load_model("models/cnnCat2.h5")

BS = 32
TS = (24, 24)
# Đánh giá trên tập test
test_batch = generator("dataset/test", shuffle=False, batch_size=BS, target_size=TS)
test_loss, test_accuracy = cnn_model.evaluate(test_batch)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Lấy một ảnh từ tập test
img, labels = next(test_batch)

# Hiển thị ảnh đầu tiên trong tập test
plt.imshow(img[20].reshape(24,24), cmap='gray')
plt.show()

# Dự đoán lớp của ảnh đầu tiên
y_predict = cnn_model.predict(img[20].reshape(1,24,24,1))
print('Giá trị dự đoán:', np.argmax(y_predict))

y_true = test_batch.classes
y_pred = cnn_model.predict(test_batch).argmax(axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create the confusion matrix plot
conf_matrix_disp = ConfusionMatrixDisplay(cm)
conf_matrix_disp.plot(cmap='Blues')