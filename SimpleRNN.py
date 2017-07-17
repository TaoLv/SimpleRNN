import theano
from theano import tensor as T
import numpy as np
import sys
import mkl_simplernn_op
from mkl_simplernn_op import SimpleRNN

np.random.seed(12345)

units = 1000    # int(sys.argv[1])
timesteps = 16  # int(sys.argv[2])
batch_size = 80 # int(sys.argv[3])
input_dim = 620 # int(sys.argv[4])

print "units=", units
print "timesteps=", timesteps
print "batch_size=", batch_size
print "input_dim=", input_dim

x = np.random.rand(timesteps, batch_size, input_dim).astype(np.float64)
w_x = np.random.rand(input_dim, units).astype(np.float64) - np.random.rand(input_dim, units).astype(np.float64)
w_h = np.random.rand(units, units).astype(np.float64) - np.random.rand(units, units).astype(np.float64)
b = np.zeros(units, dtype=np.float64)

h_init = np.zeros((batch_size, units), dtype=np.float64)


def SimpleRNN_NP():
    global x, h_init, w_x, w_h, b
    h_t = h_init

    for i in range(timesteps):
        if i != (timesteps -1) :
            h_t = np.tanh(np.dot(x[i], w_x) + np.dot(h_t, w_h) + b)
        else:
            h_t = np.dot(x[i], w_x) + np.dot(h_t, w_h) + b

    return h_t


def SimpleRNN_MKL():
    global x, h_init, w_x, w_h, b, units
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('B') 

    o = SimpleRNN(hid=units)(X, W_x, W_h, B)

    f = theano.function([X, W_x, W_h, B], o)
    theano.printing.pydotprint(f, outfile='simple_rnn.png', var_with_name_simple=True)
    o_mkl = f(x, w_x, w_h, b)
    return o_mkl


if __name__ == '__main__':
    o_numpy = SimpleRNN_NP()
    o_mkl = SimpleRNN_MKL()
    print o_numpy.shape
    print o_mkl.shape
    print(o_mkl)
    print(o_numpy)
    np.save('numpy.npy', o_numpy)
    np.save('mkl.npy', o_mkl)
    assert np.allclose(o_mkl, o_numpy)
