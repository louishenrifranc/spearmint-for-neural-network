import theano
import theano.tensor as T
from lasagne.layers import InputLayer, get_output, get_all_params, batch_norm
from lasagne.objectives import squared_error
from lasagne.updates import adadelta
import time
from utils import *


class NN():
    def __init__(self,
                 n_in,
                 layers,
                 n_epoch=100,
                 batch_size=16,
                 ):

        # PLACEHOLDERS and MODEL PARAMETERS
        self.BATCH_SIZE = batch_size
        self.X = T.fmatrix('x').astype('int8')
        self.Y = T.fvector('y')
        self.N_IN = n_in
        self.N_EPOCH = 100
        l1 = T.scalar('l1')
        l2 = T.scalar('l2')

        model = {}
        model['l_in'] = InputLayer((self.BATCH_SIZE, self.N_IN), input_var=self.X)
        for layer in layers:
            model = layer.build_layer(model, l1, l2)

        model['l_out'] = model[:-1]
        Y_hat = get_output(model['l_out'], deterministic=False)
        Y_test = get_output(model['l_out'], deterministic=True)

        all_params = get_all_params(model['l_out'], trainable=True)
        cost = T.mean(squared_error(self.Y, T.reshape(Y_hat, (Y_hat.shape[0],))), axis=0)
        loss = cost + l1 + l2
        updates = adadelta(loss, all_params)

        self.train_fn = theano.function(inputs=[self.X, self.Y], outputs=[loss], updates=updates)
        self.test_fn = theano.function(inputs=[self.X, self.Y], outputs=[cost])

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


def main(job_id, params):
    l1 = params['l1']
    # l2 = params['l2']
    n_components = params['n_comps'][0]
    return pca_reduction(n_components)


if __name__ == '__main__':
    nn = NN()
    nn.train()
