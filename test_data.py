

import theano
import numpy as np

from theano import gof
from theano import tensor as T
from theano.tensor.blas import ldflags


class mkl_tanh(gof.Op):
    __props__ = ()

    def __init__(self):
        super(mkl_tanh, self).__init__()

    def make_node(self, inp):
        x = T.as_tensor_variable(inp)

        assert x.ndim is 2

        return gof.Apply(self, [x], [x.type()])

    def c_headers(self):
        return ['<mkl.h>']

    def c_libraries(self):
        return ldflags()

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('Tanh: dtype error')

        ccode = """
            %(d)s* src = (%(d)s*) PyArray_DATA(%(x)s);

            if (%(z)s == NULL) {
                %(z)s = (PyArrayObject*) PyArray_ZEROS(2, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);
            }

            size_t total = (size_t)(PyArray_DIMS(%(x)s)[0]) * (size_t)(PyArray_DIMS(%(x)s)[1]);

            v%(dtype)sTanh(total, src, (%(d)s*)PyArray_DATA(%(z)s));
        """ % locals()

        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)


def theano_tanh(inp):
    x = T.dmatrix('x')

    z = mkl_tanh()(x)

    f = theano.function([x], z)

    return f(inp)
    

def numpy_tanh(x):
    return np.tanh(x)



if __name__ == "__main__":
    a = np.load('numpy.npy')
    b = np.load('mkl.npy')

    # assert np.allclose (a, b)

    print(a)
    print(b)
    c = theano_tanh(b)
    d = numpy_tanh(b)
    print(c)
    print(d)
    #assert np.allclose (c, d)
    e = c-d
    # ind = np.where(e == e.max())
    # print(a[ind[0][0], ind[1][0]], b[ind[0][0], ind[1][0]])
    # print(c[ind[0][0], ind[1][0]], d[ind[0][0], ind[1][0]])

    print(np.max(c-d), np.min(c-d))
