import theano
from theano import tensor as T
import numpy as np
import sys
import mkl_simplernn_op
from mkl_simplernn_op import SimpleRNN

np.random.seed(12345)

units = 1000    # int(sys.argv[1])
timesteps = 4   # int(sys.argv[2])
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

    o = SimpleRNN(hid=units, return_sequences=True)(X, W_x, W_h)
    loss = o.sum()
    gx, gwx, gwh = theano.grad(loss, [X, W_x, W_h])

    f = theano.function([X, W_x, W_h], [loss, gx, gwx, gwh])
    # theano.printing.pydotprint(f, outfile='simple_rnn_bw.png', var_with_name_simple=True)
    return f


def SimpleRNN_theano_backward():
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('B')
    Hid = T.dmatrix('hid')

    def step(x, h):
        h = T.tanh(T.dot(x, W_x) + T.dot(h, W_h))
        return h

    result, updates = theano.scan(step, sequences=[X], outputs_info=Hid, name="SimpleRNN_theano")

    loss = result.sum()
    gx, gwx, gwh  = theano.grad(loss, [X, W_x, W_h])
    f = theano.function([X, W_x, W_h, Hid], [loss, gx, gwx, gwh], updates=updates)
    # theano.printing.pydotprint(f, outfile='simple_rnn_theano.png', var_with_name_simple=True)
    return f


if __name__ == '__main__':
    f_mkl = SimpleRNN_MKL_backward()
    f_theano = SimpleRNN_theano_backward()

    a = f_mkl(x, w_x, w_h)
    b = f_theano(x, w_x, w_h, h_init)

    print('\n====Compare loss value===================')
    print('loss max err: %s' %(abs(a[0] - b[0])))
    print('loss relative err: %s' %(abs((a[0] - b[0]) / b[0])))

    print('\n====Compare gradient inputs==============')
    p, q, r = np.where((a[1]-b[1]) == (a[1]-b[1]).max())
    print('gradient inputs max err: %s' %(abs((a[1]-b[1]).max())))
    print('gradient inputs relative err: %s' %(abs((a[1]-b[1]).max()/b[1][p, q, r].max())))

    print('\n====Compare gradient weight for X========')
    p, q = np.where((a[2]-b[2]) == (a[2]-b[2]).max())
    print('gradient wegihtX max err: %s' %( abs((a[2]-b[2]).max())))
    print('gradient weightX relative err: %s' %( abs((a[2]-b[2]).max()/b[2][p, q].max())))

    print('\n====Compare gradient weight for H========')
    p, q = np.where((a[3]-b[3]) == (a[3]-b[3]).max())
    print('gradient weightH max err: %s' %( abs((a[3]-b[3]).max())))
    print('gradient weightH relative err: %s' %( abs((a[3]-b[3]).max()/b[3][p, q].max())))
    """
    print('\n====Compare gradient bias================')
    p = np.where((a[4]-b[4]) == (a[4]-b[4]).max())
    print('gradient bias max err: %s' %( abs((a[4]-b[4]).max())))
    print('gradient bias relative err: %s' %( abs((a[4]-b[4]).max()/b[4][p].max())))
    """

