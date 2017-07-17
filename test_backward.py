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


def SimpleRNN_MKL_backward():
    global x, h_init, w_x, w_h, b, units
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('B') 

    o = SimpleRNN(hid=units, return_sequences=True)(X, W_x, W_h, B)
    loss = o.sum()
    gx = theano.grad(loss, [X])

    f = theano.function([X, W_x, W_h, B], gx)
    theano.printing.pydotprint(f, outfile='simple_rnn.png', var_with_name_simple=True)
    # o_mkl = f(x, w_x, w_h, b)
    return f


def SimpleRNN_theano_backward():
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('B')
    Hid = T.dmatrix('hid')

    def step(x, h):
        h = T.nnet.sigmoid(T.dot(x, W_x) + T.dot(h, W_h) + B)
        return h

    result, updates = theano.scan(step, sequences=[X], outputs_info=Hid, name="SimpleRNN_theano")

    loss = result.sum()
    gx = theano.grad(loss, X)
    f = theano.function([X, W_x, W_h, B, Hid], gx)
    theano.printing.pydotprint(f, outfile='simple_rnn_theano.png', var_with_name_simple=True)
    return f


if __name__ == '__main__':
    # o_mkl = SimpleRNN_MKL_backward()

    o = SimpleRNN_theano_backward()
