import model, train, cv2, preproc
import numpy as np


def generate_test_data(batch_size=32):
    image_batch = np.zeros((batch_size, 66, 220, 3))
    while True:
        for i in range(batch_size):
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % i)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % i + 1)
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            image_batch[i] = diff
        yield image_batch


model = model.build_model()
model.load_weights('./weights.h5')

test_gen = train.generate_test_data()
predictions = model.predict_generator(test_gen, steps=10798)
