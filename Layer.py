from lasagne.layers import DenseLayer, DropoutLayer, batch_norm
from lasagne.regularization import regularize_layer_params_weighted, l1, l2
import utils
import json
import os
from utils import get_relative_filename

class Layer(object):
    def __init__(self,
                 layers_info):
        self.default_values = json.load(
            open(get_relative_filename('data/default_parameters_value.json'), 'rb'))
        self.non_linearity = self.get_attribute('non_linearity', layers_info)
        self.n_hidden = self.get_attribute('n_hidden', layers_info)
        self.l1_reg = self.get_attribute('l1_reg', layers_info)
        self.l2_reg = self.get_attribute('l2_reg', layers_info)
        self.dropout_p = self.get_attribute('dropout', layers_info)
        self.batch_norm = self.get_attribute('batch_norm', layers_info)
        self.name = self.get_attribute('name', layers_info)

    def get_attribute(self, name, layers_info):
        if name in layers_info:
            return layers_info[name]
        else:
            return self.default_values[name]

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
        model = DenseLayer(model,
                           num_units=self.n_hidden,
                           nonlinearity=utils.get_nonlinearity(self.non_linearity))
        if self.l1_reg != 0:
            all_l1_regs += regularize_layer_params_weighted({model: self.l1_reg}, l1)

        if self.l2_reg != 0:
            all_l2_regs += regularize_layer_params_weighted({model: self.l2_reg}, l2)

        if self.batch_norm == "Yes":
            model = batch_norm(model)
        if self.dropout_p != 0:
            model = DropoutLayer(model, p=self.dropout_p)
        return model, all_l1_regs, all_l2_regs

    def __str__(self):
        return str(
            'Layer %s: \n\tnonlinearity: %s\n\tl1 reg: %.3f\n\tl2 reg: %.3f\n\tdrop prob: '
            '%.3f\n\tnb hidden: %d\n\tbatch_norm?: %s' % (self.name, self.non_linearity, self.l1_reg, self.l2_reg,
                                                          self.dropout_p, self.n_hidden, self.batch_norm))
