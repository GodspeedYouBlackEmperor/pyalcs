from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


class Network:
    def __init__(self, input_size, action_space) -> None:
        self.model = self._build_model((input_size,), action_space)
        self.fitted = False

    def is_fitted(self):
        return self.fitted

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def fit(self, state, target, batch_size):
        self.fitted = True
        return self.model.fit(state, target, batch_size=batch_size, verbose=0)

    def _build_model(self, input_shape, action_space):
        X_input = Input(input_shape)

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size and Hidden Layer with 512 nodes
        X = Dense(512, input_shape=input_shape, activation="relu",
                  kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions
        X = Dense(action_space, activation="linear",
                  kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X)
        model.compile(loss="mse", optimizer=RMSprop(
            lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model
