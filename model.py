import theano
import theano.tensor as T
from lasagne.layers import InputLayer, get_output, get_all_params, batch_norm
from lasagne.objectives import squared_error
from lasagne.updates import adadelta
import time
from utils import *
from Layer import Layer
import json
from network_repr import *


class NN():
    def __init__(self,
                 layers,
                 parameters,
                 ):

        # PLACEHOLDERS and MODEL PARAMETERS
        self.BATCH_SIZE = parameters['batch_size']
        self.N_IN = parameters['n_in']
        self.N_EPOCH = parameters['n_epochs']

        self.X = T.fmatrix('x').astype('int8')
        self.Y = T.fvector('y')
        l1 = 0
        l2 = 0

        model = {}
        model = InputLayer((self.BATCH_SIZE, self.N_IN), input_var=self.X)
        for layer in layers:
            model, l1, l2 = layer.build_layer(model, l1, l2)

        # print(get_network_str(model))
        Y_hat = get_output(model, deterministic=False)
        Y_test = get_output(model, deterministic=True)

        all_params = get_all_params(model, trainable=True)
        cost = T.mean(squared_error(self.Y, T.reshape(Y_hat, (Y_hat.shape[0],))), axis=0)
        loss = l1 + l2 + cost
        updates = get_optimizer(parameters['optimizer'], loss, all_params, parameters['lr'], parameters['decay_lr'])

        self.train_fn = theano.function(inputs=[self.X, self.Y], outputs=[loss], updates=updates,
                                        allow_input_downcast=True, on_unused_input='ignore')
        self.test_fn = theano.function(inputs=[self.X, self.Y], outputs=[cost],
                                       allow_input_downcast=True, on_unused_input='ignore')

    def train(self):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        for epoch in range(self.N_EPOCH):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for X_batch_train in iterate_minibatches(X_train, y_train, batchsize=self.BATCH_SIZE, shuffle=True):
                err_train = self.train_fn(X_batch_train[0], X_batch_train[1])
                train_err += err_train[0]
                n_train_batches += 1

            val_err = 0
            val_rec_err = 0
            n_val_batches = 0
            for X_batch_val in iterate_minibatches(X_val, y_val, self.BATCH_SIZE, shuffle=False):
                err = self.test_fn(X_batch_val[0], X_batch_val[1])
                val_err += err[0]
                n_val_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.N_EPOCH, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / n_train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / n_val_batches))

        test_err = 0
        n_test_batches = 0
        for X_batch_test in iterate_minibatches(X_test, y_test, self.BATCH_SIZE, shuffle=False):
            err = self.test_fn(X_batch_test)
            test_err += err[0]
            n_test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / n_test_batches))
        return test_err / n_test_batches

    def set_var(layer, params, name, index, prior_values):
        layer[name] = params[name][index]
        for l in xrange(len(prior_values)):
            if prior_values[l]['layer_nb'] == index:
                if name in prior_values[l]['properties']:
                    layer[name] = prior_values[l]['properties'][name]
        return layer


def main(job_id=None, params=None):
    params = {
        'l2_reg': [1.0],
        'l1_reg': [],
        'dropout': [0.1, 0.2, 0],
        'batch_norm': ["Yes", "False", "No"],
        'non_linearity': ['relu', 'relu', 'relu'],
        'n_hiddens': []
    }

    priors = json.load(open('data/layers_specific_parameters_value.json', 'rb'))
    priors = priors['layers']

    max_depth = 0
    max_depth = max(max(len(params[param]) for param in params), len(priors))
    layers = []
    indexes = {}

    for depth in xrange(max_depth):
        layer_info = {}
        for param in params:
            if len(params[param]) > 0:
                if not param in indexes:
                    indexes[param] = 0
                if indexes[param] < len(params[param]) and indexes[param] >= 0:
                    layer_info[param] = params[param][indexes[param]]
                    indexes[param] += 1
                else:
                    indexes[param] = -1
        for layer in xrange(len(priors)):
            if priors[layer]['layer_nb'] == depth:
                for param in priors[layer]['properties']:
                    layer_info[param] = priors[layer]['properties'][param]
                    if param in indexes:
                        indexes[param] -= 1

        layer_info['name'] = 'l' + str(depth)
        layers.append(Layer(layer_info))

    parameters = get_nn_parameters()
    nn = NN(layers, parameters)
    nn.train()
    return


if __name__ == '__main__':
    main()
