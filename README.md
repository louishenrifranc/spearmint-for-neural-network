# Spearmint optimizer for MLP neural networks
* Search best architecture for standart Dense neural network
* Find the best hidden size, l1 regularization, l2 regularization, dropout probability, non linearity for every layer
* Set default values for every layer, or for each layer

# Dependencies
* lasagne, numpy, theano, cPickle, spearmint

# Optimization variables
For every layers, you can set/learn this hyperparameters:
* l1 regularization coefficient
* l2 regularization coefficient
* dropout coefficient
* set batch norm or not
* dropout coefficient
* non linearities in ['relu', 'tanh']
* number of hiddens

# Usage
No need to modify the code. Just create three files
1. Create a _config.pb_ file. Create an entry for every hyperparameters that Spearmint need to search. Notice that the size of every parameters must be the same, because every hyperparameters will be applied on every layer:
```{python}
variable {
    name = "l2_reg" 
    type = FLOAT
    size = 3 # one l2_reg will be learned for every layer
    min = 0
    max = 100
}
``` 
#### TODO : possibility of share same value at each layer, and learn globally, if it is usefull ?

2. Create a _priors.json_ file, where you can set for every layers, a predefined value for every hyperparameter. If this hyperparamer is already in the spearmint config file, it will be overwritten, and config.pb will be rewritten, to decrease the _size_ parameter of this hyperparameters.
```{json}
{
  "layers": [
    {
      "layer_nb": 1,
      "properties": {
        "l2_reg": 0.01
      }
    }
}

3. Run the script __python parser.py__. It will modify the config.file based on priors.json file.

4. If you didn't mention an hyperparameters that is learnable, you can set it a default value, that will be applied to every layer. Set default values in _default.json_
```{json}
{
  "non_linearity": "relu",
  "n_hiddens": 1000,
  "l1_reg": 0.01,
  "l2_reg": 0.02,
  "dropout": 0.2,
  "batch_norm": "True"
}
```
If you haven't defined the size of your network in _2._, a default value will be applied.
#### TODO add the possibility to choose a default depth if 1,2,3 hasn't be done.

5. Run the model with spearmint. You only need to pass it _config.pb_
#### TODO add parser for input_size, number of epochs, and batch size.


