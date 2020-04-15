import model, cv2, preproc
import numpy as np

print('loading test data')

def generate_test_data():
    while True:
        for i in range(20399):
            if i == 10797:
                i = 10796
            x1 = cv2.imread('files/train/crop/frame%d.jpg' % i)
            x2 = cv2.imread('files/train/crop/frame%d.jpg' % (i + 1))
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            diff = diff/127.5 - 1
            yield diff

model = model.build_model()
print('built model; loading weights')
model.load_weights('./weights_ck.h5')
print('predicting...')
test_gen = generate_test_data()
predictions = model.predict(test_gen, steps=20399)
print(predictions)

with open('test2.txt','w') as f:
    for i in predictions:
        f.write(str(i[0])+'\n')
        last = str(i[0])
    f.write(last+'\n')
