from lasagne.layers import DenseLayer, DropoutLayer, batch_norm
from lasagne.regularization import regularize_layer_params_weighted, l1, l2
import utils
import json


class Layer(object):
    def __init__(self,
                 layers_info):
        self.default_values = json.load(open('data/default_value.json', 'rb'))
        self.non_linearity = self.get_attribute('non_linearity', layers_info)
        self.n_hidden = self.get_attribute('n_hiddens', layers_info)
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
        model[self.name + '_hid'] = DenseLayer(model[:-1],
                                               num_units=self.n_hidden,
                                               nonlinearity=utils.get_nonlinearity(self.non_linearity))
        if self.l1_reg != 0:
            all_l1_regs += regularize_layer_params_weighted({model[:-1]: self.l1_reg}, l1)

        if self.l2_reg != 0:
            all_l2_regs += regularize_layer_params_weighted({model[:-1]: self.l2_reg}, l2)

        if self.batch_norm == "True":
            model[self.name + '_batch_norm'] = batch_norm(model[:-1])
        if self.dropout_p != 0:
            model[self.name + '_drop'] = DropoutLayer(model[:-1], p=self.dropout_p)
        return model, all_l1_regs, all_l2_regs
