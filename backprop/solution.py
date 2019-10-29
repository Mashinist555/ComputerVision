from interface import *


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values,
                n - batch size, ... - arbitrary input shape
        :return: np.array((n, ...)), output values,
                n - batch size, ... - arbitrary output shape (same as input)
        """
        # your code here \/

        result = np.copy(inputs)
        result[result < 0] = 0
        return result
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                n - batch size, ... - arbitrary output shape
        :return: np.array((n, ...)), dLoss/dInputs,
                n - batch size, ... - arbitrary input shape (same as output)
        """
        # your code here \/
        inputs = self.forward_inputs
        result = np.zeros(grad_outputs.shape)
        result[inputs > 0] = 1
        return result * grad_outputs
        # your code here /\


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of units
        :return: np.array((n, d)), output values,
                n - batch size, d - number of units
        """
        # your code here \/
        exps = np.exp(inputs - np.max(inputs))
        sums = np.expand_dims(np.sum(exps, axis=1), axis=1)
        return exps / sums
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                n - batch size, d - number of units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of units
        """
        # your code here \/
        outputs = self.forward_outputs
        batch_size = grad_outputs.shape[0]
        size = grad_outputs.shape[1]
        along_i = np.repeat(outputs[:, :, np.newaxis], size, axis=2)
        along_j = np.repeat(outputs[:, np.newaxis, :], size, axis=1)
        din_out = along_i * np.repeat(np.eye(size)[np.newaxis, :, :], batch_size, axis=0) * along_j
        # inputs = self.forward_inputs
        # exps = np.exp(inputs - np.max(inputs))
        # sums = np.expand_dims(np.sum(exps, axis=1), axis=1)
        # sum_except = sums - exps
        result = np.tensordot(din_out, grad_outputs, axes=[1, 2])

        return outputs * sum_except / sums * grad_outputs
        # your code here /\


# =============================== 2.1.3 Dense ================================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def forward(self, inputs):
        """
        :param inputs: np.array((n, d)), input values,
                n - batch size, d - number of input units
        :return: np.array((n, c)), output values,
                n - batch size, c - number of output units
        """
        # your code here \/
        batch_size, input_units = inputs.shape
        output_units, = self.output_shape
        return np.empty((batch_size, output_units))
        # your code here /\

    def backward(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                n - batch size, c - number of output units
        :return: np.array((n, d)), dLoss/dInputs,
                n - batch size, d - number of input units
        """
        # your code here \/
        batch_size, output_units = grad_outputs.shape
        input_units, = self.input_shape
        inputs = self.forward_inputs

        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad[...] = np.empty(self.weights.shape)
        # dLoss/dBiases
        self.biases_grad[...] = np.empty(self.biases.shape)

        return np.empty((batch_size, input_units))
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n,)), loss scalars for batch
        """
        # your code here \/
        batch_size, output_units = y_gt.shape
        return np.empty((batch_size,))
        # your code here /\

    def gradient(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n, d)), gradient loss to y_pred
        """
        # your code here \/
        return np.empty(y_pred.shape)
        # your code here /\


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            return np.empty(parameter_shape)
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            assert parameter_shape == updater.inertia.shape

            # Don't forget to update the current inertia tensor:
            updater.inertia[...] = np.empty(parameter_shape)
            return np.empty(parameter_shape)
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = ...
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(...)
    # 3) Train and validate the model using the provided data
    model.fit(...)
    # your code here /\
    return model

# ============================================================================
