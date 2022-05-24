from p2pfl.learning.model import MyNodeLearning
from p2pfl.agregator import FedAvg    
from collections import OrderedDict
import torch


def test_encoding():
    nl1 = MyNodeLearning(None)
    params = nl1.encode_parameters()

    nl2 = MyNodeLearning(None)
    nl2.set_parameters(nl2.decode_parameters(params))

    params == nl2.encode_parameters()


def test_avg_simple():

    a = OrderedDict([('a', -1), ('b', -1)])
    b = OrderedDict([('a', 0), ('b', 0)])
    c = OrderedDict([('a', 1), ('b', 1)])

    result = FedAvg.agregate([a,b,c])

    for layer in b:
        assert result[layer] == b[layer]

def test_avg_complex():
    nl1 = MyNodeLearning(None)
    params = nl1.get_parameters()
    params1 = nl1.get_parameters()
    params2 = nl1.get_parameters()

    result = FedAvg.agregate([params])

    # Check Results
    for layer in params:
        assert torch.eq(params[layer], result[layer]).all()

    for layer in params2:
        params1[layer] = params1[layer]+1
        params2[layer] = params2[layer]-1
    
    result = FedAvg.agregate([params1.copy(), params2.copy()])


    # Check Results
    for layer in params:
        a = torch.round(params[layer], decimals=2)
        b = torch.round(result[layer], decimals=2)
        print(torch.eq(a, b).all())
