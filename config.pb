language: PYTHON
name:     "model"

variable{
    name = "n_hiddens"
    type = FLOAT
    size = 3
    min = 0
    max = 100
}
variable {
    name = "l1_reg"
    type = FLOAT
    size = 3
    min = 0
    max = 100
}

variable {
    name = "l2_reg"
    type = FLOAT
    size = 3
    min = 0
    max = 100
}

variable {
    name = "dropout"
    type = FLOAT
    size = 3
    min = 0
    max = 1
}

variable {
    name = "batch_norm"
    type = ENUM
    size = 3
    options = "True"
    options = "False"
}

variable {
    name = "non_linearity"
    type = ENUM
    size = 3
    options = "relu"
    options = "tanh"
}