import model, cv2, preproc


def generate_test_data():
    while True:
        for i in range(10798):
            if i == 10797:
                i = 10796
            x1 = cv2.imread('files/test/crop/frame%d.jpg' % i)
            x2 = cv2.imread('files/test/crop/frame%d.jpg' % (i + 1))
            diff = preproc.optical_flow(x1, x2)
            diff = diff.reshape(1, diff.shape[0], diff.shape[1], diff.shape[2])
            diff = diff/127.5 - 1
            yield diff


model = model.build_model()
model.load_weights('./weights_ck.h5')
test_gen = generate_test_data()
predictions = model.predict(test_gen, steps=10798)

with open('test.txt', 'w') as f:
    for i in predictions:
        f.write(str(i[0])+'\n')
