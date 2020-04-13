from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Nadam
import numpy as np
import model
import preproc
import cv2
import matplotlib.pyplot as plt

with open('files/train.txt') as f:
    speeds = f.read().splitlines()
speeds = np.array(speeds).astype(np.float)


def generate_training_data(batch_size=16):
    image_batch = np.zeros((batch_size, 66, 220, 3))
    label_batch = np.zeros(batch_size)
    fr = 4080
    while True:
        for i in range(batch_size):
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % fr)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % (fr+1))
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            y = np.mean([speeds[fr], speeds[fr+1]])

            image_batch[i] = diff
            label_batch[i] = y
            fr += 1
        shuffle(image_batch, label_batch)
        yield (image_batch, label_batch)


def generate_validation_data():
    while True:
        for i in range(4080):
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % i)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % (i + 1))
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            y = np.mean([speeds[i], speeds[i + 1]])
            speed = np.array([[y]])
            yield (diff, speed)


model = model.build_model()
adam = Nadam()
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

print(model.summary())
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=2,
                              verbose=1,
                              min_delta=0.23,
                              mode='min', )
modelCheckpoint = ModelCheckpoint('weights.h5',
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1,
                                  save_weights_only=True)
callbacks_list = [modelCheckpoint, earlyStopping]
train_generator = generate_training_data(32)
test_generator = generate_validation_data()
history = model.fit(train_generator,
                    steps_per_epoch=508,
                    epochs=25,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_data=test_generator)

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
