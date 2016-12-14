from lasagne.layers import DenseLayer, DropoutLayer, batch_norm
from lasagne.regularization import regularize_layer_params_weighted, l1, l2


class Layer(object):
    def __init__(self,
                 non_linearity,
                 n_hidden,
                 l1_reg,
                 l2_reg,
                 dropout_p,
                 batch_norm,
                 name):
        self.non_linearity = non_linearity
        self.n_hidden = n_hidden
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.name = name

    def name(self):
        return self.name

    def non_linerity(self):
        return self.non_linearity

    def n_hiddens(self):
        return self.n_hidden

    def l1_reg(self):
        return self.l1_reg

    def l2_reg(self):
        return self.l2_reg

    def dropout_probs(self):
        return self.dropout_p

    def is_batch_norm(self):
        return self.batch_norm

    def build_layer(self, model, all_l1_regs, all_l2_regs):
        model[self.name + '_hid'] = DenseLayer(model[:-1],
                                               num_units=self.n_hidden,
                                               nonlinearity=self.non_linearity)
        if self.l1_reg != 0:
            all_l1_regs += regularize_layer_params_weighted({model[:-1]: self.l1_reg}, l1)

        if self.l2_reg != 0:
            all_l2_regs += regularize_layer_params_weighted({model[:-1]: self.l2_reg}, l2)

        if self.batch_norm == "True":
            model[self.name + '_batch_norm'] = batch_norm(model[:-1])
        if self.dropout_p != 0:
            model[self.name + '_drop'] = DropoutLayer(model[:-1], p=self.dropout_p)
        return model, all_l1_regs, all_l2_regs
