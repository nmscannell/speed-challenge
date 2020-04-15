from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import model
import preproc
import cv2
import matplotlib.pyplot as plt

with open('files/train.txt') as f:
    speeds = f.read().splitlines()
speeds = np.array(speeds).astype(np.float)

images = np.zeros((20399, 66, 220, 3))
labels = np.zeros(20399)

for i in range(20399):
    print(i)
    x1 = cv2.imread('files/train/crop/frame%d.jpg' % i)
    x2 = cv2.imread('files/train/crop/frame%d.jpg' % (i+1))
    diff = preproc.optical_flow(x1, x2)
    diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
    diff = diff/127.5 - 1
    y = np.mean([speeds[i], speeds[i+1]])
    images[i] = diff
    labels[i] = y
images, labels = shuffle(images, labels)


model = model.build_model()

earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=2,
                              verbose=1,
                              min_delta=0.23,
                              mode='min', )
modelCheckpoint = ModelCheckpoint('weights_ck.h5',
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1,
                                  save_weights_only=True)
callbacks_list = [modelCheckpoint, earlyStopping]

history = model.fit(images, labels,
                    shuffle=True,
                    epochs=25,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_split=.2)

model.save_weights('weights2.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(25)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('plots.png')
