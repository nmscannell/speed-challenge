from tensorflow.python.keras.models import load_model
import model, train, cv2, preproc


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

'''
def predict(data, model):
    results = []
    for i in range(10798):
        x1 = cv2.imread('files/test/crop/frame%d.jpg' % i)
        x2 = cv2.imread('files/test/crop/frame%d.jpg' % i + 1)
        diff = preproc.optical_flow(x1, x2)
        diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
        y = np.mean([speeds[i], speeds[i + 1]])

        pred = model.predict(diff)
        error = abs(pred)
'''


model = model.build_model()
model.load_weights('./weights.h5')

test_gen = train.generate_test_data()
predictions = model.predict_generator(test_gen, steps=10798)
