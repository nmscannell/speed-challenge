from sklearn.utils import shuffle
import numpy as np
import model
import preproc
import cv2

with open('files/train.txt') as f:
    speeds = f.read().splitlines()


def generate_training_data(batch_size=16):
    image_batch = np.zeros((batch_size, 66, 220, 3))
    label_batch = np.zeros(batch_size)
    fr = 3060
    while True:
        for i in range(batch_size):
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % fr)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % fr+1)
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            y = np.mean([speeds[fr], speeds[fr+1]])

            image_batch[i] = diff
            label_batch[i] = y
            fr += 1

        yield shuffle(image_batch, label_batch)


def generate_validation_data():
    while True:
        for i in range(3060):
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % i)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % i + 1)
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            y = np.mean([speeds[i], speeds[i + 1]])
            speed = np.array([[y]])
            yield diff, speed


model = model.build_model()
#    dl_model = model.speed_model()
#    earlyStopping = EarlyStopping(monitor='val_loss',
#                                  patience=2,
#                                  verbose=1,
#                                  min_delta=0.23,
#                                  mode='min', )
#    modelCheckpoint = ModelCheckpoint(filepath,
#                                      monitor='val_loss',
#                                      save_best_only=True,
#                                      mode='min',
#                                      verbose=1,
#                                      save_weights_only=True)
#    callbacks_list = [modelCheckpoint, earlyStopping]
train_generator = generate_training_data()
test_generator = generate_validation_data()
history = model.fit_generator(
        train_generator,
        steps_per_epoch=555,
        epochs=25,
#        callbacks=callbacks_list,
        verbose=1,
        validation_data=test_generator)
