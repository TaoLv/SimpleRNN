import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags
from mkl_simplernn_backward import SimpleRNNGradInputs

class SimpleRNN(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences')

    def __init__(self, hid, step=None, dim=None, return_sequences=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        super(SimpleRNN, self).__init__()

    def make_node(self, *inputs):
        if len(inputs) in (3, 4):
            inp = list(map(tensor.as_tensor_variable, inputs))
        else:
            raise ValueError('SimpleRNN: number of parameter is wrong.')

        assert inp[0].ndim is 3
        assert inp[1].ndim is 2
        assert inp[2].ndim is 2

        if self.return_sequences:
            out = [inp[0].type()]
        else:
            bcast = [inp[0].type.broadcastable[1], inp[0].type.broadcastable[2]]
            out = [tensor.tensor(dtype=inp[0].type.dtype, broadcastable=bcast)]

        return gof.Apply(self, inp, out)

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
            %(dtype)s** A;
            %(dtype)s** B;
            %(dtype)s** C;

            MKL_INT    m_g[1];
            MKL_INT    k_g[1];
            MKL_INT    n_g[1];
            MKL_INT    lda_g[1];
            MKL_INT    ldb_g[1];
            MKL_INT    ldc_g[1];

            CBLAS_TRANSPOSE    transA_g[1];
            CBLAS_TRANSPOSE    transB_g[1];

            %(dtype)s  alpha_g[1];
            %(dtype)s  beta_g[1];
            MKL_INT    size_per_grp[1];

            size_t time_step;
            size_t batch_size;
            size_t embed_dims;

            %(dtype)s* temp;
            %(dtype)s* hid_state;
            """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
            A = NULL;
            B = NULL;
            C = NULL;

            m_g[0] = 0;
            k_g[0] = 0;
            n_g[0] = 0;

            lda_g[0] = 0;
            ldb_g[0] = 0;
            ldc_g[0] = 0;

            transA_g[0] = CblasNoTrans;
            transB_g[0] = CblasNoTrans;

            alpha_g[0] = 1.0;
            beta_g[0] = 1.0;
            size_per_grp[0] = 1;

            time_step = 0;
            batch_size = 0;
            embed_dims = 0;

            temp = NULL;
            hid_state = NULL;
            """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (A) {
                mkl_free (A);
                A = NULL;
            }

            if (B) {
                mkl_free (B);
                B = NULL;
            }

            if (C) {
                mkl_free (C);
                C =NULL;
            }

            if (hid_state) {
                mkl_free(hid_state);
                hid_state = NULL;
            }

            if (temp) {
                mkl_free(temp);
                temp = NULL;
            }
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        if len(inputs) is 3:
            with_bias = 0
            x, w_x, w_h = inputs
        elif len(inputs) is 4:
            with_bias = 1
            x, w_x, w_h, b = inputs
        else:
            raise TypeError('SimpleRNN: too much arguments')

        z, = outputs
        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

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
            time_step  = PyArray_DIMS(%(x)s)[0];
            batch_size = PyArray_DIMS(%(x)s)[1];
            embed_dims = PyArray_DIMS(%(x)s)[2];

            %(d)s* x_ptr = NULL;
            %(d)s* w_x_ptr = NULL;
            %(d)s* w_h_ptr = NULL;
            %(d)s* b_ptr = NULL;

            if (A == NULL)
                A = (%(d)s**)mkl_malloc(time_step * sizeof (%(d)s*), 64);

            if (B == NULL)
                B = (%(d)s**)mkl_malloc(time_step * sizeof (%(d)s*), 64);

            if (C == NULL)
                C = (%(d)s**)mkl_malloc(time_step * sizeof (%(d)s*), 64);

            // Cast input arrays to C contiguous style to do GEMM
            PyArrayObject* x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(x)s)) {
                printf(\"Warning: Need convert x to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(x)s,
                                                PyArray_TYPE(%(x)s),
                                                PyArray_NDIM(%(x)s),
                                                PyArray_NDIM(%(x)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"SimpleRNN: fail to cast x to C-Contiguous array\");
                    goto simplernn_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(x)s);
            }

            PyArrayObject* w_x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(w_x)s)) {
                printf(\"Warning: Need convert w_x to C-Contiguous\\n\");
                w_x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(w_x)s,
                                                PyArray_TYPE(%(w_x)s),
                                                PyArray_NDIM(%(w_x)s),
                                                PyArray_NDIM(%(w_x)s));
                if (!w_x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"SimpleRNN: fail to cast w_x to C-Contiguous array\");
                    goto simplernn_fail;
                }
                w_x_ptr = (%(d)s*) PyArray_DATA(w_x_src);
            } else {
                w_x_ptr = (%(d)s*) PyArray_DATA(%(w_x)s);
            }

            PyArrayObject* w_h_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(w_h)s)) {
                printf(\"Warning: Need convert w_h to C-Contiguous\\n\");
                w_h_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(w_h)s,
                                                PyArray_TYPE(%(w_h)s),
                                                PyArray_NDIM(%(w_h)s),
                                                PyArray_NDIM(%(w_h)s));
                if (!w_h_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"SimpleRNN: fail to cast w_h to C-Contiguous array\");
                    goto simplernn_fail;
                }
                w_h_ptr = (%(d)s*) PyArray_DATA(w_h_src);
            } else {
                w_h_ptr = (%(d)s*) PyArray_DATA(%(w_h)s);
            }

            // temp is used to store output of dot(x, w_x)
            if (NULL == temp) {
                temp = (%(d)s*) mkl_malloc(time_step * batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            """ % locals()

        if with_bias:
            ccode += """
            b_ptr = (%(d)s*) PyArray_DATA(%(b)s);
            #pragma omp parallel for
            for (int i = 0; i < time_step; i++) {
                for (int j = 0; j < batch_size; j++) {
                size_t offset = %(hid)s * j + %(hid)s * batch_size * i;
                    memcpy((void*)(temp + offset), (void*)b_ptr, %(hid)s * sizeof(%(d)s));
                }
            }
            """ % locals()
        else:
            ccode += """
            memset((char*)temp, 0, time_step * batch_size * %(hid)s * sizeof (%(d)s));
            """ % locals()

        ccode += """
            m_g[0] = batch_size;
            k_g[0] = embed_dims;
            n_g[0] = %(hid)s;
            lda_g[0] = k_g[0];
            ldb_g[0] = n_g[0];
            ldc_g[0] = n_g[0];
            size_per_grp[0] = time_step;

            for (int i = 0 ; i < time_step; i++) {
                A[i] = x_ptr + i * m_g[0] * k_g[0];
                B[i] = w_x_ptr;
                C[i] = temp + i * m_g[0] * n_g[0];
            }
            // Batch Gemm for dot(x, w_x) + b
            cblas_%(dtype)sgemm_batch(CblasRowMajor, transA_g, transB_g, m_g, n_g, k_g,
                                      alpha_g, A, lda_g, B, ldb_g, beta_g, C, ldc_g, 1, size_per_grp);

            // construct output
            npy_intp dims[3] = {0, 0, 0};
            if (NULL == %(z)s) {
                if (%(return_sequences)s) {
                    dims[0] = time_step;
                    dims[1] = batch_size;
                    dims[2] = %(hid)s;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(x)s), 0);
                } else {
                    dims[0] = batch_size;
                    dims[1] = %(hid)s;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(x)s), 0);
                }
            }
 
            if (NULL == %(z)s) {
                PyErr_SetString(PyExc_RuntimeError, \"SimpleRNN: create output array failed\");
                goto simplernn_fail;
            }

            // init hidden state
            if (NULL == hid_state) {
                hid_state = (%(d)s*) mkl_malloc(batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            if (NULL == hid_state) {
                PyErr_SetString(PyExc_MemoryError, \"SimpleRNN: create buffer for hidden state failed\");
                goto simplernn_fail;
            }

            memset((char*)hid_state, 0, batch_size * %(hid)s * sizeof (%(d)s));

            size_t size_per_batch = batch_size * %(hid)s * sizeof (%(d)s);
            for (int i = 0; i < time_step; i++) {
                // Gemm for dot(h_tm1, w_h) + temp
                %(d)s* p = temp + i * batch_size * %(hid)s;
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, %(hid)s, %(hid)s,
                                    1.0, hid_state, %(hid)s, w_h_ptr, %(hid)s, 1.0, p, %(hid)s);

                if (i != (time_step - 1))
                    v%(dtype)sTanh(batch_size * %(hid)s, temp + i * batch_size * %(hid)s, hid_state);
                else
                    memcpy ((char*)hid_state, (char*)p, batch_size * %(hid)s * sizeof (%(d)s));
                if (%(return_sequences)s) {
                    memcpy(((char*)PyArray_DATA(%(z)s)) + i * size_per_batch,
                            (char*)hid_state,
                            size_per_batch);
                } else {
                    if (i == time_step - 1) {
                        memcpy(((char*)PyArray_DATA(%(z)s)),
                                (char*)hid_state,
                                size_per_batch);
                    }
                }
            }

            simplernn_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(w_x_src);
            Py_XDECREF(w_h_src);

            """ % locals()
        return ccode

    def grad(self, inp, grads):
        x, w_x, w_h, = inp[0:3]
        gz, = grads
        hid = SimpleRNN(self.hid, self.step, self.dim, self.return_sequences)(*inp)
        gradX = SimpleRNNGradInputs(self.hid, self.step, self.dim, self.return_sequences)(w_x, w_h, hid, gz)

        if len(inp) is 3:
            return [gradX, gradX, gradX]
        else:
            return [gradX, gradX, gradX, gradX]

    def c_code_cache_version(self):
        return (1, 0, 0)
