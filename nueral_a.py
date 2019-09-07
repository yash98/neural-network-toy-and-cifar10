import sys
import numpy as np
import math


class value_func:
    @classmethod
    def func(cls, x):
        raise NotImplementedError

    @classmethod
    def derivative(cls, x):
        raise NotImplementedError

    @classmethod
    def should_exist(cls, *args):
        raise NotImplementedError

    @classmethod
    def matrix_func(cls, x):
        raise NotImplementedError

    @classmethod
    def matrix_derivative(cls, x):
        raise NotImplementedError


class matrix_func:
    @classmethod
    def func(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def derivative(cls, *args, **kwargs):
        raise NotImplementedError


class layer:
    def __init__(self, width, weight_dimensions, activation):
        self.width = width
        self.weight_dimensions = weight_dimensions
        self.activation = activation
        self.weights = np.zeros(weight_dimensions)
        self.grad_wrt_w = np.zeros(weight_dimensions)
        self.acc_grad_wrt_w = np.zeros(weight_dimensions)
        self.z_s = np.zeros((1, width))
        self.a_s = np.zeros((1, width))
        self.delta = np.zeros((1, width))

    def clear_acc_grad_wrt_w(self):
        self.acc_grad_wrt_w = np.zeros(self.weight_dimensions)


class nueral_network:

    """
    layers contain hidden layers + output layer
    activations is list of class inherited from value_func
    lose_func is a class inherited from matrix_func
    """

    def __init__(self, layer_widths, activations, input_size, output_size, loss_func):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.loss_func = loss_func
        self.input = None
        self.output = None

        # make layers
        for i in range(len(layer_widths)):
            # (n^k-1 + 1, n^k)
            weight_dimensions = (0, 0)
            if i == 0:
                weight_dimensions = (input_size+1, layer_widths[i])
            else:
                weight_dimensions = (layer_widths[i-1]+1, layer_widths[i])

            self.layers.append(
                layer(layer_widths[i], weight_dimensions, activations[i]))

    """
    x is a column vector without 1 appended at start
    """

    def forward_prop(self, x):
        self.input = x
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[0].z_s = (np.append(np.ones((1, 1)), x,
                                                axis=1)) @ self.layers[0].weights
                self.layers[0].a_s = self.layers[0].activation.matrix_func(
                    self.layers[0].z_s)
            else:
                self.layers[i].z_s = (np.append(np.ones((1, 1)), self.layers[i-1].a_s,
                                                axis=1)) @ self.layers[i].weights
                self.layers[i].a_s = self.layers[i].activation.matrix_func(
                    self.layers[i].z_s)

        self.output = self.layers[-1].a_s
        return self.output

    def back_prop(self, y, **kwargs):
        # not sure if correct for general loss function
        for i in range(len(self.layers)-1, -1, -1):
            # delta calculation
            if i == len(self.layers)-1:
                if getattr(self.loss_func, "last_delta_calc", None) is not None:
                    dE_by_dz = self.loss_func.last_delta_calc(self.output, y)
                else:
                    dE_by_da = self.loss_func.derivative(
                        self.output, y, **kwargs)
                    dE_by_dz = dE_by_da * \
                        self.layers[-1].activation.matrix_derivative(
                            self.layers[-1].z_s)
                self.layers[-1].delta = dE_by_dz
            else:
                dE_by_dz = self.layers[i].activation.matrix_derivative(
                    self.layers[i].z_s) * (self.layers[i+1].delta @ (self.layers[i+1].weights[1:, ]).T)
                self.layers[i].delta = dE_by_dz

            # grad wrt weight calculation
            if i == 0:
                self.layers[0].grad_wrt_w = (
                    np.append(np.ones((1, 1)), self.input, axis=1)).T @ self.layers[0].delta
                self.layers[0].acc_grad_wrt_w += self.layers[0].grad_wrt_w
            else:
                self.layers[i].grad_wrt_w = (
                    np.append(np.ones((1, 1)), self.layers[i - 1].a_s, axis=1)).T @ \
                    self.layers[i].delta
                self.layers[i].acc_grad_wrt_w += self.layers[i].grad_wrt_w

    def clear_acc_grad_wrt_w(self):
        for layer in self.layers:
            layer.clear_acc_grad_wrt_w()

    def unlearn(self):
        for layer in self.layers:
            layer.weights = np.zeros(layer.weight_dimensions)
            layer.grad_wrt_weight = np.zeros(layer.weight_dimensions)
            layer.z_s = np.zeros((1, layer.width))
            layer.a_s = np.zeros((1, layer.width))
            layer.delta = np.zeros((1, layer.width))

        self.input = None
        self.output = None
        self.learning_rate = None

    def output_weights(self):
        l = []
        for layer in self.layers:
            l.append(layer.weights)
        return l

    def print_weights(self, outputfile_name):
        outputfile = open(outputfile_name, "w")
        for layer in self.layers:
            for i in range(layer.weight_dimensions[1]):
                for j in range(layer.weight_dimensions[0]):
                    outputfile.write(str(layer.weights[j][i])+"\n")
        outputfile.close()

    def fixed_rate_mini_batch_gd(self, x, y, learning_rate, batch_size, max_epoch, **kwargs):
        num_epoch = 0
        # last_error = 0.0
        # current_error = 0.0
        # curr_w = None
        # last_w = None
        pause_c = 0
        while num_epoch < max_epoch:
            # last_error = current_error
            # last_w = curr_w
            # current_error = 0.0

            # relevent input for loss function
            d = {"batch_size": batch_size}

            for i in range((num_epoch*batch_size) % x.shape[0], ((num_epoch+1)*batch_size) % x.shape[0]):
                self.forward_prop(x[i:i+1, :])
                # current_error += (-1/float(batch_size))*self.loss_func.func(
                #     curr_w, y)
                self.back_prop(y[i:i+1, :], **d)

            for layer in self.layers:
                layer.weights -= (1/float(batch_size)) * \
                    learning_rate*layer.acc_grad_wrt_w
                layer.clear_acc_grad_wrt_w()

            if "pauses" in kwargs:
                if pause_c < len(kwargs["pauses"]):
                    if num_epoch == (kwargs["pauses"])[pause_c]:
                        self.print_weights(
                            kwargs["pausefile_prefix"]+str((kwargs["pauses"])[pause_c]))
                        pause_c += 1

            num_epoch += 1

        return num_epoch


class sigmoid(value_func):
    @classmethod
    def func(cls, x):
        return 1 / (1+math.exp(-x))

    @classmethod
    def derivative(cls, x):
        sigma = cls.func(x)
        return sigma*(1-sigma)

    @classmethod
    def matrix_func(cls, x):
        r = np.zeros(x.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r[i][j] = sigmoid.func(x[i][j])
        return r

    @classmethod
    def matrix_derivative(cls, x):
        r = np.zeros(x.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r[i][j] = sigmoid.derivative(x[i][j])
        return r


class bce_loss(matrix_func):
    @classmethod
    def func(cls, v, y, **kwargs):
        return np.array([[-1*(y[0][0]*np.log(v[0][0]) + (1-y[0][0])*np.log(1-v[0][0])), ], ])

    @classmethod
    def derivative(cls, v, y, **kwargs):
        if y[0][0] == 1:
            return np.array([[-1*(1/v[0][0]), ], ])
        else:
            return np.array([[-1*(1/(1-v[0][0])), ], ])

    @classmethod
    def last_delta_calc(cls, v, y):
        return np.array([[v[0][0] - y[0][0], ], ])


if __name__ == "__main__":
    # python nueral_a.py Toy/train.csv Toy/param.txt weightfile
    trainfile_name = sys.argv[1]
    param_name = sys.argv[2]
    weightfile_name = sys.argv[3]

    param = open(param_name, "r")
    param_lines = param.readlines()

    widths = [int(s) for s in param_lines[4:]]
    widths.append(1)
    activation_l = [sigmoid for i in range(len(widths))]
    nn = nueral_network(widths, activation_l, 2, 1, bce_loss)

    inp = np.loadtxt(trainfile_name, delimiter=",")
    x = inp[:, :-1]
    y = inp[:, -1:]

    if (int(param_lines[0])) == 1:
        print(nn.fixed_rate_mini_batch_gd(
            x, y, float(param_lines[1]), int(param_lines[3]), int(param_lines[2]), pauses=[1, 10, 20, 30, 40, 50], pausefile_prefix="itera"))

    nn.print_weights(weightfile_name)
    param.close()
