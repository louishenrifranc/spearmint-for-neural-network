from lasagne.layers import DenseLayer, DropoutLayer, batch_norm
from lasagne.regularization import regularize_layer_params_weighted, l1, l2
import utils


class Layer(object):
    def __init__(self,
                 layers_info):
        self.non_linearity = layers_info['non_linearity']
        self.n_hidden = layers_info['n_hidden']
        self.l1_reg = layers_info['l1_reg']
        self.l2_reg = layers_info['l2_reg']
        self.dropout_p = layers_info['dropout']
        self.batch_norm = layers_info['batch_norm']
        self.name = layers_info['name']

    def name(self):
        return self.name

    def non_linearity(self):
        return self.non_linearity

    def n_hidden(self):
        return self.n_hidden

    def l1_reg(self):
        return self.l1_reg

    def l2_reg(self):
        return self.l2_reg

    def dropout_prob(self):
        return self.dropout_p

    def is_batch_norm(self):
        return self.batch_norm

    def build_layer(self, model, all_l1_regs, all_l2_regs):
        model = DenseLayer(model,
                           num_units=self.n_hidden,
                           nonlinearity=utils.get_non_linearity(self.non_linearity))
        if self.l1_reg != 0:
            all_l1_regs += regularize_layer_params_weighted({model: self.l1_reg}, l1)

        if self.l2_reg != 0:
            all_l2_regs += regularize_layer_params_weighted({model: self.l2_reg}, l2)

        if self.batch_norm == "Y":
            model = batch_norm(model)
        if self.dropout_p != 0:
            model = DropoutLayer(model, p=self.dropout_p)
        return model, all_l1_regs, all_l2_regs

    def __str__(self):
        return str(
            'Layer %s: \n\tnonlinearity: %s\n\tl1 reg: %.3f\n\tl2 reg: %.3f\n\tdrop prob: '
            '%.3f\n\tnb hidden: %d\n\tbatch_norm?: %s' % (self.name, self.non_linearity, self.l1_reg, self.l2_reg,
                                                          self.dropout_p, self.n_hidden, self.batch_norm))
