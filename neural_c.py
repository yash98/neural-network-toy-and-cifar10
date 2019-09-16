import sys
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder


class value_func:
    @classmethod
    def val_func(cls, x):
        raise NotImplementedError

    @classmethod
    def val_derivative(cls, x):
        raise NotImplementedError

    @classmethod
    def should_exist(cls, *args):
        raise NotImplementedError

    @classmethod
    def func(cls, x):
        raise NotImplementedError

    @classmethod
    def derivative(cls, x):
        raise NotImplementedError


class matrix_func:
    @classmethod
    def func(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def derivative(cls, *args, **kwargs):
        raise NotImplementedError


class layer:
    def __init__(self, width, weight_dimensions, activation, weight_init):
        self.width = width
        self.weight_dimensions = weight_dimensions
        self.activation = activation
        if weight_init == "zero":
            self.weights = np.zeros(weight_dimensions)
            self.biases = np.zeros((1, width))
        elif weight_init == "random":
            self.weights = np.random.rand(
                weight_dimensions[0], weight_dimensions[1])
            self.biases = np.random.rand(1, width)
        self.grad_wrt_w = np.zeros(weight_dimensions)
        self.grad_wrt_b = np.zeros((1, width))
        self.z_s = None
        self.a_s = None


class neural_network:

    """
    layers contain hidden layers + output layer
    activations is list of class inherited from value_func
    lose_func is a class inherited from matrix_func
    """

    def __init__(self, layer_widths, activations, input_size, output_size, loss_func, weight_init):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.loss_func = loss_func
        self.input = None
        self.output = None
        self.weight_init = weight_init

        # make layers
        for i in range(len(layer_widths)):
            # (n^k-1, n^k)
            weight_dimensions = (0, 0)
            if i == 0:
                weight_dimensions = (input_size, layer_widths[i])
            else:
                weight_dimensions = (layer_widths[i-1], layer_widths[i])

            self.layers.append(
                layer(layer_widths[i], weight_dimensions, activations[i], weight_init))

    """
    x is a column vector without 1 appended at start
    """

    def forward_prop(self, x):
        self.input = x
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[0].z_s = (
                    self.input @ self.layers[0].weights) + self.layers[0].biases
                self.layers[0].a_s = self.layers[0].activation.func(
                    self.layers[0].z_s)
            else:
                self.layers[i].z_s = (
                    self.layers[i-1].a_s @ self.layers[i].weights) + self.layers[i].biases
                self.layers[i].a_s = self.layers[i].activation.func(
                    self.layers[i].z_s)

        self.output = self.layers[-1].a_s
        return self.output

    def back_prop(self, y, **kwargs):
        # not sure if correct for general loss function
        last_delta = None
        current_delta = None
        for i in range(len(self.layers)-1, -1, -1):
            # delta calculation
            if i == len(self.layers)-1:
                if getattr(self.loss_func, "last_delta_calc", None) is not None:
                    dE_by_dz = self.loss_func.last_delta_calc(self.output, y)
                else:
                    dE_by_da = self.loss_func.derivative(
                        self.output, y, **kwargs)
                    dE_by_dz = dE_by_da * \
                        self.layers[-1].activation.derivative(
                            self.layers[-1].z_s)
                current_delta = dE_by_dz
            else:
                dE_by_dz = np.multiply(self.layers[i].activation.derivative(
                    self.layers[i].z_s), (last_delta @ (self.layers[i+1].weights).T))
                current_delta = dE_by_dz

            # grad wrt weight calculation
            if i == 0:
                self.layers[0].grad_wrt_w = self.input.T @ current_delta
                self.layers[0].grad_wrt_b = ((np.ones(
                    (self.input.shape[0], 1))).T) @ current_delta
            else:
                self.layers[i].grad_wrt_w = self.layers[i -
                                                        1].a_s.T @ current_delta
                self.layers[i].grad_wrt_b = ((np.ones(
                    (self.input.shape[0], 1))).T) @ \
                    current_delta
            last_delta = current_delta

    def unlearn(self):
        for layer in self.layers:
            if self.weight_init == "zero":
                self.weights = np.zeros(layer.weight_dimensions)
                self.biases = np.zeros((1, layer.width))
            elif self.weight_init == "random":
                self.weights = np.random.rand(
                    layer.weight_dimensions[0], layer.weight_dimensions[1])
                self.biases = np.random.rand(1, layer.width)
            layer.grad_wrt_weight = np.zeros(layer.weight_dimensions)
            self.grad_wrt_b = np.zeros((1, layer.width))
            layer.z_s = np.zeros((1, layer.width))
            layer.a_s = np.zeros((1, layer.width))

        self.input = None
        self.output = None
        self.learning_rate = None

    def output_weights(self):
        l = []
        for layer in self.layers:
            l.append(layer.weights)
            l.append(layer.biases)
        return l

    def input_weights(self, weight_list):
        i = 0
        for layer in self.layers:
            layer.weights = weight_list[i]
            i += 1
            layer.biases = weight_list[i]
            i += 1

    def print_weights(self, outputfile_name):
        out_w = []
        for layer in self.layers:
            for bs in layer.biases:
                for b in bs:
                    out_w.append(b)
            for ws in layer.weights:
                for w in ws:
                    out_w.append(w)
        np.savetxt(outputfile_name, out_w)

    def mini_batch_gd(self, x, y, learning_rate, batch_size, max_epoch, learning_strat, **kwargs):
        num_epoch = 0
        num_batch = 0
        total_batches = x.shape[0]//batch_size
        test_x = x[(total_batches-1)*batch_size:total_batches*batch_size, :]
        test_y = y[(total_batches-1)*batch_size:total_batches*batch_size, :]
        x = x[:(total_batches-1)*batch_size, :]
        y = y[:(total_batches-1)*batch_size, :]
        total_batches = (x.shape[0]//batch_size)
        least_error = (np.sum(self.loss_func.func(
            self.forward_prop(test_x), test_y), axis=0))
        best_w = self.output_weights()
        pause_c = 0
        while num_epoch < max_epoch:
            self.forward_prop(
                x[(num_batch*batch_size):((num_batch+1)*batch_size), :])

            self.back_prop(
                y[(num_batch*batch_size):((num_batch+1)*batch_size), :])

            curr_error = (np.sum(self.loss_func.func(self.forward_prop(
                test_x), test_y, ), axis=0))

            if (curr_error < least_error):
                least_error = curr_error
                best_w = self.output_weights()

            num_epoch += 1
            num_batch = (num_batch+1) % total_batches

            if learning_strat == "fixed":
                for layer in self.layers:
                    layer.weights -= (learning_rate * layer.grad_wrt_w)
                    layer.biases -= (learning_rate * layer.grad_wrt_b)
            elif learning_strat == "adaptive":
                for layer in self.layers:
                    layer.weights -= ((learning_rate /
                                       math.sqrt(num_epoch)) * layer.grad_wrt_w)
                    layer.biases -= ((learning_rate /
                                      math.sqrt(num_epoch)) * layer.grad_wrt_b)

            if "pauses" in kwargs:
                if pause_c < len(kwargs["pauses"]):
                    if num_epoch == (kwargs["pauses"])[pause_c]:
                        self.print_weights(
                            kwargs["pausefile_prefix"]+str((kwargs["pauses"])[pause_c]))
                        pause_c += 1

        self.input_weights(best_w)
        return (least_error, best_w)


class sigmoid(value_func):
    @classmethod
    def val_func(cls, x):
        return 1 / (1+math.exp(-x))

    @classmethod
    def val_derivative(cls, x):
        sigma = cls.func(x)
        return sigma*(1-sigma)

    @classmethod
    def func(cls, x):
        z = np.clip(x, -500, 500)
        return 1 / (1+np.exp(-z))

    @classmethod
    def derivative(cls, x):
        z = np.clip(x, -500, 500)
        sig = sigmoid.func(z)
        return sig*(1-sig)


class bce_loss(matrix_func):
    @classmethod
    def func(cls, v, y):
        return -1*(y*np.log(v) + (1-y)*np.log(1-v))

    @classmethod
    def derivative(cls, v, y):
        return (((1-y)/(1-v))-(y/v))

    @classmethod
    def last_delta_calc(cls, v, y):
        return (v - y) / v.shape[0]


class softmax(matrix_func):
    @classmethod
    def func(cls, x):
        ex = np.exp(x)
        return (ex.T / np.sum(ex, axis=1)).T

    # softmax cant be applied to intermediatary layers
    # so no derivative, derivative of 1d becomes 2d


class cross_entropy_loss(matrix_func):
    @classmethod
    def func(cls, v, y):
        return (-1/v.shape[0])*(np.sum(v*y, axis=0))

    # not implementing right now
    # @classmethod
    # def derivative(cls, v, y):
    #     return (((1-y)/(1-v))-(y/v))

    @classmethod
    def last_delta_calc(cls, v, y):
        return (v - y) / v.shape[0]


class tan_h:
    @classmethod
    def func(cls, x):
        return np.tanh(x)

    @classmethod
    def derivative(cls, x):
        return 1-np.square(np.tanh(x))


if __name__ == "__main__":
    # python neural_a.py Neural_data/Toy/train.csv Neural_data/Toy/param.txt weightfile
    trainfile_name = sys.argv[1]
    testfile_name = sys.argv[2]
    outputfile_name = sys.argv[3]

    widths = [100, 20]
    widths.append(10)
    activation_l = [tan_h for i in range(len(widths)-1)]
    activation_l.append(softmax)
    nn = neural_network(widths, activation_l, 1024, 10,
                        cross_entropy_loss, "random")

    traindata = np.loadtxt(trainfile_name, delimiter=",")
    x = traindata[:, :-1]
    x = (1.0/255.0)*x
    y = traindata[:, -1:]
    y_enc = OneHotEncoder(handle_unknown='ignore', n_values=10)
    y_enc.fit(y)
    y = y_enc.transform(y).toarray()

    bw = None
    le = 0.0
    for i in range(10):
        (le1, bw1) = nn.mini_batch_gd(x, y, 0.5, 100, 1000, "adaptive")
        if (le1 < le):
            bw = bw1
        nn.unlearn()
    nn.input_weights(bw)

    testdata = np.loadtxt(testfile_name, delimiter=",")
    test_out = nn.forward_prop((1.0/255.0)*testdata[:, :-1])

    final_out = []
    for ohv in test_out:
        i = 0
        max_v = ohv[0]
        max_vi = 0
        for val in ohv:
            if (val > max_v):
                max_v = val
                max_vi = i
            i += 1
        final_out.append(max_vi)

    np.savetxt(outputfile_name, final_out)
