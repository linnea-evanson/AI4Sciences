import numpy as np
from linear_reg import linear_regression

def test_input_type(x):
    x = np.random.randn(3,2)
    assert isinstance(x, np.ndarray)
        # print("Input data type correct")

def test_output_type(y):
    y = np.random.randn(3)
    assert isinstance(y, np.ndarray)
        # print("Output data type correct")

def test_dimensions():
    x = np.random.randn(3,2)
    y = np.random.randn(3)
    coef = linear_reg(x,y)
    assert len(coef) == 2