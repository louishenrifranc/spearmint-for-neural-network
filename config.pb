language: PYTHON
name:     "model"


variable {
    name : "l1_reg"
    type : FLOAT
    size : 2
    min : 0
    max : 2
}

variable {
    name : "l2_reg"
    type : FLOAT
    size : 2
    min : 0
    max : 2
}

variable {
    name : "dropout"
    type : FLOAT
    size : 2
    min : 0
    max : 1
}

