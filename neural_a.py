import sys
import numpy as np
import math
# from decimal import Decimal


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
        self.biases = np.zeros((1, width))
        self.grad_wrt_w = np.zeros(weight_dimensions)
        self.grad_wrt_b = np.zeros((1, width))
        # self.acc_grad_wrt_w = None
        self.z_s = None
        self.a_s = None
        self.delta = None

    # def clear_grad_wrt_w(self):
    #     self.grad_wrt_w = np.zeros(self.weight_dimensions)


class neural_network:

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
                weight_dimensions = (input_size, layer_widths[i])
            else:
                weight_dimensions = (layer_widths[i-1], layer_widths[i])

            self.layers.append(
                layer(layer_widths[i], weight_dimensions, activations[i]))

    """
    x is a column vector without 1 appended at start
    """

    def forward_prop(self, x):
        self.input = x
        for i in range(len(self.layers)):
            if i == 0:
                # a = (np.append(np.ones((x.shape[0], 1)), x, axis=1))
                # b = self.layers[0].weights
                # self.layers[0].z_s = a @ b
                self.layers[0].z_s = (
                    self.input @ self.layers[0].weights) + self.layers[0].biases
                self.layers[0].a_s = self.layers[0].activation.matrix_func(
                    self.layers[0].z_s)
            else:
                # a = (np.append(np.ones(
                #     (self.layers[i-1].a_s.shape[0], 1)), self.layers[i-1].a_s, axis=1))
                # b = self.layers[i].weights
                # self.layers[i].z_s = a @ b
                self.layers[i].z_s = (
                    self.layers[i-1].a_s @ self.layers[i].weights) + self.layers[i].biases
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
                a = self.layers[i].activation.matrix_derivative(
                    self.layers[i].z_s)
                b = self.layers[i+1].delta
                c = (self.layers[i+1].weights).T
                dE_by_dz = a * (b @ c)
                self.layers[i].delta = dE_by_dz

            # grad wrt weight calculation
            if i == 0:
                # a = (
                #     np.append(np.ones((self.input.shape[0], 1)), self.input, axis=1)).T
                # b = self.layers[0].delta
                self.layers[0].grad_wrt_w = self.input.T @ self.layers[0].delta
                self.layers[0].grad_wrt_b = ((np.ones(
                    (self.input.shape[0], 1))).T) @ self.layers[0].delta
            else:
                # a = (np.append(np.ones(
                #     (self.layers[i - 1].a_s.shape[0], 1)), self.layers[i - 1].a_s, axis=1)).T
                # b = self.layers[i].delta
                # self.layers[i].grad_wrt_w = a @ b
                self.layers[i].grad_wrt_w = self.layers[i -
                                                        1].a_s.T @ self.layers[i].delta
                self.layers[i].grad_wrt_b = ((np.ones(
                    (self.input.shape[0], 1))).T) @ \
                    self.layers[i].delta

    # def clear_grad_wrt_w(self):
    #     for layer in self.layers:
    #         layer.clear_grad_wrt_w()

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
        # outputfile = open(outputfile_name, "w")
        out_w = []
        for layer in self.layers:
            for bs in layer.biases:
                for b in bs:
                    out_w.append(b)
            for ws in layer.weights:
                for w in ws:
                    out_w.append(w)
        np.savetxt(outputfile_name, out_w)
        # outputfile.close()

    def mini_batch_gd(self, x, y, learning_rate, batch_size, max_epoch, learning_strat, **kwargs):
        num_epoch = 0
        num_batch = 0
        total_batches = x.shape[0]//batch_size
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
            # d = {"batch_size": batch_size}

            self.forward_prop(
                x[(num_batch*batch_size):((num_batch+1)*batch_size), :])
            # current_error += (-1/float(batch_size))*self.loss_func.func(
            #     curr_w, y)
            self.back_prop(
                y[(num_batch*batch_size):((num_batch+1)*batch_size), :])

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

            num_epoch += 1
            num_batch = (num_batch+1) % total_batches

            if "pauses" in kwargs:
                if pause_c < len(kwargs["pauses"]):
                    if num_epoch == (kwargs["pauses"])[pause_c]:
                        self.print_weights(
                            kwargs["pausefile_prefix"]+str((kwargs["pauses"])[pause_c]))
                        pause_c += 1

        return num_epoch


class sigmoid(value_func):
    @classmethod
    def func(cls, x):
        return 1 / (1+math.exp(-x))

    @classmethod
    def derivative(cls, x):
        sigma = cls.func(x)
        return sigma*(1-sigma)
        # return np.exp(-x) / np.square((1+np.exp(-x)))

    @classmethod
    def matrix_func(cls, x):
        # r = np.zeros(x.shape)
        # for i in range(r.shape[0]):
        #     for j in range(r.shape[1]):
        #         r[i][j] = sigmoid.func(x[i][j])
        # return r
        return 1 / (1+np.exp(-x))

    @classmethod
    def matrix_derivative(cls, x):
        # r = np.zeros(x.shape)
        # for i in range(r.shape[0]):
        #     for j in range(r.shape[1]):
        #         r[i][j] = sigmoid.derivative(x[i][j])
        # return r
        return np.exp(-x) / (np.square((1+np.exp(-x))))


class bce_loss(matrix_func):
    @classmethod
    def func(cls, v, y, **kwargs):
        # r = np.zeros(v.shape)
        # for i in range(v.shape[0]):
        #     for j in range(v.shape[1]):
        #         r[i][j] = -1*(y[i][j]*np.log(v[i][j]) +
        #                       (1-y[i][j])*np.log(1-v[i][j]))
        # return r
        return -1*(y*np.log(v) + (1-y)*np.log(1-v))

    @classmethod
    def derivative(cls, v, y, **kwargs):
        # r = np.zeros(v.shape)
        # for i in range(v.shape[0]):
        #     for j in range(v.shape[1]):
        #         if y[i][j] == 1:
        #             r[i][j] = -1/(v[i][j])
        #         else:
        #             r[i][j] = 1/(1-v[i][j])
        # return r
        return (((1-y)/(1-v))-(y/v))

    @classmethod
    def last_delta_calc(cls, v, y):
        # r = np.zeros(v.shape)
        # for i in range(v.shape[0]):
        #     for j in range(v.shape[1]):
        #         r[i][j] = v[i][j] - y[i][j]
        # return r
        return (v - y) / v.shape[0]


if __name__ == "__main__":
    # python neural_a.py Neural_data/Toy/train.csv Neural_data/Toy/param.txt weightfile
    trainfile_name = sys.argv[1]
    param_name = sys.argv[2]
    weightfile_name = sys.argv[3]

    param = open(param_name, "r")
    param_lines = param.readlines()

    widths = [int(s) for s in param_lines[4:]]
    widths.append(1)
    activation_l = [sigmoid for i in range(len(widths))]
    nn = neural_network(widths, activation_l, 2, 1, bce_loss)

    inp = np.loadtxt(trainfile_name, delimiter=",")
    x = inp[:, :-1]
    y = inp[:, -1:]

    if (int(param_lines[0])) == 1:
        print(nn.mini_batch_gd(x, y, float(param_lines[1]), int(param_lines[3]),
                               int(param_lines[2]), "fixed", pauses=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
                               pausefile_prefix="itera"))
        # print(nn.mini_batch_gd(x, y, float(param_lines[1]), int(param_lines[3]),
        #                        int(param_lines[2]), "fixed"))
    elif (int(param_lines[0])) == 2:
        # print(nn.mini_batch_gd(x, y, float(param_lines[1]), int(param_lines[3]),
        #                        int(param_lines[2]), "adaptive", pauses=[1, 10, 20, 30, 40, 50],
        #                        pausefile_prefix="itera"))
        print(nn.mini_batch_gd(x, y, float(param_lines[1]), int(param_lines[3]),
                               int(param_lines[2]), "adaptive"))

    nn.print_weights(weightfile_name)
    param.close()
