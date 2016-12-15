language: PYTHON
name:     "model"


variable {
    name : "l1_reg"
    type : FLOAT
    size : 3
    min : 0
    max : 100
}

variable {
    name : "l2_reg"
    type : FLOAT
    size : 3
    min : 0
    max : 100
}

variable {
    name : "dropout"
    type : FLOAT
    size : 3
    min : 0
    max : 1
}
