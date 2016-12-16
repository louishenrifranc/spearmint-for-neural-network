# Spearmint optimizer for MLP neural networks
* Search best hyperparameters for neural network.
* Find best hyperparameters for every dense layer in the neural network. Specify which parameter will be learned, and set the other a specific value.
* No code, only json, and pb files

# Dependencies
* Install dependencies with pip
```{bash}
pip install -r requirements.txt
```
* Install Lasagne from github 
```{bash}
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```
* Install spearmint. More info ![here](https://github.com/JasperSnoek/spearmint)

# Optimization variables
Each layer in the neural network has different hyperparameters from the number of hidden neuron to the l1 regularization applied on its weights.  
From the first hidden layer, up to the output layer,parameters can be learned/pre-defined (globally or indenpendantly). A layer is defined as:
```
class Layer:
	"""
	Params
	------	
	l1_reg: float
		l1 regularization coefficient	
	l2_reg: float
		l2 regularization coefficient
	drop_p: float (between 0 and 1)
		dropout probablity
	batch_norm: boolean ("Yes", "False")
		batch normalize, or not, before activation
	non_linearity: string ("relu", "tanh")
		activation function 
	n_hidden: integer
		number of hidden neurons
	# more to come
``` 

# Usage

## Create a _config.pb_ file. 
Create an entry for every hyperparameters that you want _Spearmint_ to learn. _Size_ parameter should be the depth of your neural network

```{python}
# Example for the l2_reg parameter
variable {
    name = "l2_reg" 
    type = FLOAT
    size = 3 # number of layers = 3
    min = 0
    max = 100
}
```

## Create a _predefined_values.json_ file.
Every parameter of any Layer, can be set to a value. It prevent _Spearmint_ to learn it. __Unless the number of hidden neurons is a parameter to learn, make sure to set it a value. If you don't, a default value from the default_values.json will be affected to every layer__.  

#### How to structure your json file
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

## Run the script __parser.py__. 
It will modify the config.file based on predefined_values.json file. Parser script is in the script folder

## Create a __default_values.json file__
If a parameter of a layer is not to learn, and hadn't either be manually set, give it a default value, in this file.
```{json}
{
  "non_linearity": "relu",
  "n_hidden": 1000,
  "l1_reg": 0.01,
  "l2_reg": 0.02,
  "dropout": 0.2,
  "batch_norm": "True"
}
```

## Modify global parameters, such as n_epochs, n_inputs, batchsize, optimizer in __global_nn_parameters.json__

## Run spearmint
* Run spearmint from the spearmint bin/ folder with the command:
```{bash}
./spearmint path_to_project_folder/config.pb  --driver=local --method=GPEIChooser --method-args=noiseless=0 --max-concurrent=2
```
* Cleanup the project folder
```{bash}
./cleanup path_to_project_folder
```
More info about _Spearmint_ on ![here](https://github.com/JasperSnoek/spearmint)



