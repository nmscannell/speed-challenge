from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Nadam

IMG_SHAPE = (66, 220, 3)


def build_model():
    inputs = Input(shape=IMG_SHAPE)
    inputs1 = Lambda(lambda x: x/127.5 - 1, input_shape=IMG_SHAPE)(inputs)

    conv1 = Conv2D(24, (5, 5), padding='valid', activation='relu')(inputs1)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu')(conv1)
    drop1 = Dropout(0.5)(conv2)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu')(drop1)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    max1 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(64, (3, 3), padding='valid', activation='relu')(max1)

    flat1 = Flatten()(conv5)
    dense1 = Dense(128, activation='relu')(flat1)
    dense2 = Dense(64, activation='relu')(dense1)
    dense4 = Dense(16, activation='relu')(dense2)
    output = Dense(1, activation='relu')(dense4)

    model = Model(inputs, output)
    adam = Nadam()
    model.compile(optimizer=adam, loss='mse')

    print(model.summary())
    return model
