# Spearmint optimizer for MLP neural networks
* Search best hyperparameter for standart Dense neural network
* Learn any hyperparameter of any layers, or set it a default value.
* Using only protobuf file, and json files

# Dependencies
* lasagne, numpy, theano, cPickle, spearmint

# Optimization variables
Each layer in the neural network, from the first hidden, to the output one, has parameters that can be learned/pre-defined/shared with other layer as a global constant. A layer is defined as:
```
class Layer:
	"""
	Params
	------	
	l1_reg: float
		l1 regularization coefficient	
	l2 reg: float
		l2 regularization coefficient
	drop_p: float (between 0 and 1)
		dropout probablity
	batch_norm: boolean ("Yes", "False")
		batch normalize, or not, before activation
	non linearity: string ("relu", "tanh")
		activation function 
	number of hiddens: integer
		number of hidden neurons
	# more to come
``` 

# Usage
No need to modify the code. Just modify three files	

## Create a _config.pb_ file. 
Create an entry for every hyperparameters that you want _Spearmint_ to learn. Make sure the size parameter is the umber of layers
```{python}
# Example for the l2_reg parameter
variable {
    name = "l2_reg" 
    type = FLOAT
    size = 3 # number of layers = 3
    min = 0
    max = 100
}
#### TODO : possibility of share same value at each layer, and learn globally, if it is usefull ?
``` 


## Create a _priors.json_ file.
Every parameter of any Layer, can be set to a predefined value, even if it was supposed to be learned by _Spearmint_. __Unless the number of hiddens is a learned parameters, make sure to set its value. If you don't, a default value with be affected to every layer__. 
1. "layer_nb" : the layers depth, started at 0
2. "properties" : you can set every parameter from the _Layer_ object.
#### Example
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
```
## Run the script __python parser.py__. 
It will modify the config.file based on priors.json file.

## Create a __default.json file__
If a parameter of a layer is not to learn neither manually set at any depth, give it a default value, in this file.
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

## Modify global parameters, such as n_epochs, n_inputs, batchsize, optimizer

## Run spearmint
* Run spearmint from the spearmint bin/ folder with the command:
```{bash}
./spearmint ../examples/path_to_project_folder/config.pb  --driver=local --method=GPEIChooser --method-args=noiseless=0 --max-concurrent=2
```
* Cleanup the project folder
```{bash}
./spearmint ../examples/path_to_project_folder
```
More info on ![spearmint github page](https://github.com/JasperSnoek/spearmint)



