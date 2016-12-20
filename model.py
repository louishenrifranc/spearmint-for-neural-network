import theano.tensor as T
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.objectives import squared_error
import time
from theano import function
import sys

from script.utils import get_optimizer, load_dataset, iterate_minibatches, get_layers, get_nn_parameters


class NN():
    def __init__(self,
                 layers,
                 parameters):

        # Global parameter of the neural network
        self.BATCH_SIZE = parameters['batch_size']
        self.N_IN = parameters['n_in']
        self.N_EPOCH = parameters['n_epochs']

        # Input and output
        self.X = T.fmatrix('x').astype('int8')
        self.Y = T.fvector('y')
        l1, l2 = 0, 0

        # Input layer
        model = InputLayer((self.BATCH_SIZE, self.N_IN), input_var=self.X)
        # Stack layer
        for layer in layers:
            model, l1, l2 = layer.build_layer(model, l1, l2)

        # Get output for training and testing phase
        Y_hat = get_output(model, deterministic=False)
        Y_test = get_output(model, deterministic=True)

        # Get all weighs
        all_params = get_all_params(model, trainable=True)

        # Cost functions
        cost = T.mean(squared_error(self.Y, T.reshape(Y_hat, (Y_hat.shape[0],))), axis=0)
        cost_test = T.mean(squared_error(self.Y, T.reshape(Y_test, (Y_test.shape[0],))), axis=0)

        # Loss function
        loss = l1 + l2 + cost

        # Optimizer
        updates = get_optimizer(parameters['optimizer'], loss, all_params, parameters['lr'], parameters['decay_lr'])

        # Theano functions
        self.train_fn = function(inputs=[self.X, self.Y], outputs=[loss], updates=updates,
                                 allow_input_downcast=True, on_unused_input='ignore')
        self.test_fn = function(inputs=[self.X, self.Y], outputs=[cost_test],
                                allow_input_downcast=True, on_unused_input='ignore')

    def train(self):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        for epoch in range(self.N_EPOCH):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for X_batch_train in iterate_minibatches(X_train, y_train, self.BATCH_SIZE, shuffle=True):
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
            err = self.test_fn(X_batch_test[0], X_batch_test[1])
            test_err += err[0]
            n_test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / n_test_batches))
        return test_err / n_test_batches


def main(job_id=None, params=None):
    # Get the layers
    layers = get_layers(params)
    # Get neural network defined function
    parameters = get_nn_parameters()
    # Build the neural network
    nn = NN(layers, parameters)
    # Train the neural network
    return nn.train()


if __name__ == '__main__':
    nn = NN((), ())
