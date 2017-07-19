import theano
from theano import tensor as T
import numpy as np
import sys
import mkl_simplernn_op
from mkl_simplernn_op import SimpleRNN
import time

np.random.seed(12345)

units = 1000    # int(sys.argv[1])
timesteps = 4  # int(sys.argv[2])
batch_size = 80 # int(sys.argv[3])
input_dim = 620 # int(sys.argv[4])

print "units=", units
print "timesteps=", timesteps
print "batch_size=", batch_size
print "input_dim=", input_dim

x = np.random.rand(timesteps, batch_size, input_dim).astype(np.float32)
w_x = np.random.rand(input_dim, units).astype(np.float32) - np.random.rand(input_dim, units).astype(np.float32)
w_h = np.random.rand(units, units).astype(np.float32) - np.random.rand(units, units).astype(np.float32)
b = np.zeros(units, dtype=np.float32)

h_init = np.zeros((batch_size, units), dtype=np.float32)


def SimpleRNN_NP():
    global x, h_init, w_x, w_h, b
    h_t = h_init

    tic = time.time()
    for i in range(timesteps):
        h_t = np.tanh(np.dot(x[i], w_x) + np.dot(h_t, w_h))
    toc = time.time()
    print('NumPy Time: %.8f' %(toc - tic))
    return h_t


def SimpleRNN_MKL():
    global x, h_init, w_x, w_h, b, units
    X = T.ftensor3('X')
    W_x = T.fmatrix('W_x')
    W_h = T.fmatrix('W_h')
    B = T.fvector('B') 

    o = SimpleRNN(hid=units, return_sequences=True)(X, W_x, W_h)

    f = theano.function([X, W_x, W_h], o)
    # theano.printing.pydotprint(f, outfile='simple_rnn.png', var_with_name_simple=True)
    o_mkl = f(x, w_x, w_h)
    
    """
    tic = time.time()
    for i in range(1000):
        o_mkl = f(x, w_x, w_h)
    toc = time.time()
    print('MKL Time: %.8f' %((toc-tic)/1000))
    """
    return o_mkl


def SimpleRNN_theano():
    X = T.ftensor3('X')
    W_x = T.fmatrix('W_x')
    W_h = T.fmatrix('W_h')
    B = T.fvector('B')
    Hid = T.fmatrix('hid')

    def step(x, h):
        h = T.tanh(T.dot(x, W_x) + T.dot(h, W_h))
        return h

    result, updates = theano.scan(step, sequences=[X], outputs_info=Hid, name="SimpleRNN_theano")
    f = theano.function([X, W_x, W_h, Hid], result)
    o_theano = f(x, w_x, w_h, h_init)
    """
    tic = time.time()
    for i in range(1000):
        o_theano = f(x, w_x, w_h, h_init)
    toc = time.time()
    print('Theano time: %.8f' %((toc - tic) / 1000))
    """
    return o_theano


if __name__ == '__main__':
    o_numpy = SimpleRNN_NP()
    o_mkl = SimpleRNN_MKL()
    o_theano = SimpleRNN_theano()
    # assert np.allclose(o_mkl, o_theano)
    err = o_mkl - o_theano
    print(err.max())
    print(np.where(err==err.max()))
    # print(o_mkl)
    # print(o_theano)
    # print(o_numpy)
    """
    print o_numpy.shape
    print o_mkl.shape
    print(o_mkl)
    print(o_numpy)
    np.save('numpy.npy', o_numpy)
    np.save('mkl.npy', o_mkl)
    assert np.allclose(o_mkl, o_numpy)
    """
