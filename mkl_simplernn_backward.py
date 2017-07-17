
import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class SimpleRNNGradInputs(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences')

    def __init__(self, hid, step=None, dim=None, return_sequences=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        super(SimpleRNNGradInputs, self).__init__()

    def make_node(self, W_x, W_h, hid_state, grads):
        W_x = tensor.as_tensor_variable(W_x)
        W_h = tensor.as_tensor_variable(W_h)
        hid_state = tensor.as_tensor_variable(hid_state)
        gz = tensor.as_tensor_variable(grads)

        out = [hid_state.type()]

        return gof.Apply(self, [W_x, W_h, hid_state, gz], out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype is 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype is 'float64':
            dtype = 'double'
        else:
            raise TypeError('SimpleRNN: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """

        size_t time_step;
        size_t batch_size;
        size_t hid_dims;
        size_t embed_dims;

        PyArrayObject* dhnext;

        %(dtype)s* temp;
            """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        time_step = 0;
        batch_size = 0;
        hid_dims = 0;
        embed_dims = 0;

        dhnext = NULL;
        temp = NULL;
        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """

        if (temp) {
            mkl_free (temp);
            temp = NULL;
        }
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        w_x, w_h, hid_state, gz = inputs
        gx, = outputs

        hid = self.hid
        fail = sub['fail']
        if node.inputs[0].type.dtype is 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype is 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('SimpleRNN: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))
        # print locals()

        ccode = """
            time_step  = PyArray_DIMS(%(hid_state)s)[0];
            batch_size = PyArray_DIMS(%(hid_state)s)[1];
            hid_dims   = PyArray_DIMS(%(hid_state)s)[2];

            embed_dims = PyArray_DIMS(%(w_x)s)[1];

            npy_intp dims[3] = {0, 0, 0};

            dims[0] = batch_size;
            dims[1] = hid_dims;

            if (NULL == dhnext) {
                dhnext = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(hid_state)s), 0);
            }

            %(d)s* dy_ptr = (%(d)s*) PyArray_DATA(%(gz)s);
            %(d)s* dhnext_ptr = (%(d)s*) PyArray_DATA(dhnext);
            %(d)s* h_ptr = (%(d)s*) PyArray_DATA(%(hid_state)s);

            if (NULL == temp) {
                temp = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == %(gx)s) {
                dims[0] = time_step;
                dims[1] = batch_size;
                dims[2] = embed_dims;
                %(gx)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(hid_state)s), 0);
            }

            if (NULL == %(gx)s) {
                PyErr_SetString(PyExc_RuntimeError, \"SimpleRNNGradInputs: create output array failed\");
                %(fail)s;
            }

            for (int i = time_step - 1; i >= 0; i--) {
                // dh = dy + dhnext
                v%(dtype)sAdd(batch_size * hid_dims, dy_ptr, dhnext_ptr, dhnext_ptr);
                // h(t)^2
                v%(dtype)sSqr(batch_size * hid_dims, h_ptr + i * batch_size * hid_dims, temp);
                // 1-h(t)^2
                for (int t = 0; t < batch_size * hid_dims; t++) {
                    temp[i] = 1.0 - temp[i];
                }
                // dhraw = dh * (1 - h(t)^2)
                v%(dtype)sMul(batch_size * hid_dims, temp, dhnext_ptr, temp);
                // dx = dot(dhram, W.T)
                %(d)s* gx_ptr = (%(d)s*)PyArray_DATA(%(gx)s) + i * batch_size * embed_dims;
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, embed_dims, hid_dims,
                                    1.0, temp, hid_dims, (%(d)s*) PyArray_DATA(%(w_x)s), embed_dims, 0.0, gx_ptr, embed_dims);

                // dhnext=dot(U.T, dhraw)
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, temp, hid_dims, (%(d)s*) PyArray_DATA(%(w_h)s), hid_dims, 0.0, dhnext_ptr, hid_dims);
            }

            """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)
