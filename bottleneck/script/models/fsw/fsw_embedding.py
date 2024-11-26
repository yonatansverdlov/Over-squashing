# Fourier Sliced-Wasserstein Embedding for Multisets and Measures

# Authors: Tal Amir, Nadav Dym
# Technion Institute of Technology
# Haifa, Israel

version = '2.14'
version_date = '2024-09-30'

# Edge features:
# - Do not split self.projVecs. Do self.projVecs.shape[1] == d_in + d_edge.
#
# Conditions:
# - Edge features are used iff d_edge > 0
# - Edge features are only allowed in graph mode
# - An input X_edge is expected in forward(). Its default is None, and a tensor with numel()==0 is also treated as None.
# - X_edge.shape == (<W.shape>, d_edge)
#   X_edge is sparse iff W is sparse. If sparse, its dense dimension must equal 1, and its nonzero pattern must match that of W. It must be coalesced (as is W).

# TODO:
# 0. Support d_in=0
# 1. If d_in+d_edge=1, all proj vecs should equal 1.
# 2. Add support for zero-sized output along any possible dimension (flat input corresponding to empty multisets, d_out=0)
# 3. Accelerate by explicitly computing W, W_sum and W_cumsum when W='uniform' or 'unit'
# 3. Rename 'cartesian_mode' to 'cartesian_freqs'
# 3. Add support for edge features and cartesian mode together, or at least add warning that it doesn't work
# 4. Replace torch's sparse tensors by custom handling of indices, values
# 5. For safety, in all functions under the class sp that return sparse tensors, make sure gradient of input is off
# 6. Make sure that the state_dict saves all the embedding parameters, not just the pytorch tensors (e.g. encode_total_mass).
# 7. Implement custom .state_dict() and .load_state_dict() methods, which should save and load accompanyting non-tensor parameters,
#    upon loading update required_grads of tensors according to .learnable_projs, .learnable_freqs, and allow initializing a model
#    directly from a saved state_dict.

# Changelog:
# 2.13    added support for total_mass_encoding_scale
# 2.12    added support for total_mass_encoding_function
# 2.11    fixed sparse padding bug
# 2.1     added full edge-feature support
# 2.09a   edge feature support is working (beta)
# 2.03a   can handle failed loading of custom CUDA library
# 2.02a   more efficient gradient handling in permute_sparse; renamed dimension variable names d, m to d_in, d_out
# 2.01a   added safety mechanisms to detect uncoanesced tensors
# 2.0a    low total-mass padding; total-mass encoding
# 1.6     finished optimizing code
# 1.54    speed up: 11 sec. / epoch; removed outdated code
# 1.53    speed up: 12 sec. / epoch; before removing outdated code pieces
# 1.52    speed up: 13 sec. / epoch
# 1.51    Speed up: 17 sec. / epoch
# 1.5     Speed up by sum_sparse, incorporated segcumsum
# 1.44    Added end-to-end support for int64 indexing
# 1.43a   Testing the computation time of sum(A,dim=0) when A is sparse
# 1.42    More memory-efficient sparse_cumsum backward()
# 1.41    Added reverse option to sparse_cumsum
# 1.4     Incorporated segcumsum
# 1.31b   Reverted to correct & slow sparse cumsum due to bug in the new segcumsum
# 1.31a   Testing hierarchical segcumsum
# 1.30    Added CUDA-implemented segcumsum, still slow
# 1.29    Added CUDA-implemented segcumsum!
# 1.28t3  Added cumsum_segments_consecutive
# 1.28t2  Testing cumsum_segments using jit
# 1.28t1  Testing sparse_cumsum_alt1
# 1.27    Made some slow assersions run only when fsw_embedding_debug_mode or fsw_embedding_basic_safety_checks are True
# 1.26    Removed the sp_old class. Got rid of more coalesce()
# 1.25c   Got rid of some more coalesce()
# 1.25b   Got rid of some more coalesce()
# 1.25    Removed most of coalesce() calls that follow calls to torch.sparse_coo_tensor()
# 1.24    Removed attributes 'device' and 'dtype' from FSW_embedding due to lack of safety. Use get_device(), get_dtype() instead.
# 1.23    Added project_W()
# 1.22    Added support for biases
#         learnable_freqs=True now initializes frequencies to zero
# 1.21    Added reset_parameters()
#         Added type hinting and enforcement



import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd.function import once_differentiable

import warnings
import numbers
import type_enforced # Note: A runtime error in this line implies that that some function below is given an input arguemnt of the wrong type
import importlib.util

import sys
import os
import time

import ctypes

# Load the segcumsum shared library from the same directory as the current file
mydir = os.path.dirname(os.path.abspath(__file__))
libfsw_embedding_path = os.path.join(mydir, "cuda/libfsw_embedding.so")
# This library will be loaded at FSW_embedding.__init__()
libfsw_embedding = None
#libfsw_embedding = ctypes.CDLL(libfsw_embedding_path)


# Turn this on to run some verifications and sanity checks during runtime.
# If an error is encountered, a runtime error is raised
fsw_embedding_debug_mode = False

# Conduct basic safety checks, mainly on the user input.
# Recommended to leave True, unless running time is of utmost importance, and the input is known to be consistent.
# Setting this to False does not significantly reduce running time.
fsw_embedding_basic_safety_checks = True

# Tells whether to use float64 in numerically-challenging parts of the code even if the data is in float32 format.
# This was not observed to increase accuracy, and it incurs a significantly narrower memory bottleneck and a slightly higher running time.
fsw_embedding_high_precision = False

# Produce a runtime error (rather than a warning) when failing to load the custom CUDA library
fsw_embedding_produce_error_on_custom_library_loading_failure = True

tal_global_timer = 0
tal_global_timer_start = 0

''' Maps multisets in R^d_in to vectors in R^d_out using the Fourier Sliced-Wasserstrin Embedding.
    Also supports weighted point-clouds in R^d_in, which are regarded as discrete distributions over R^d_in.
    
    The Euclidean distance between two embeddings approximates the Sliced-Wasserstein distance
    distance the input distributions:
                ||embed(X1,W1)-embed(X2,W2)||_2  =<approx>=  sqrt(d_out) * SW((X1,W1),(X2,W2))

    To guarantee that the embedding is injective with at most n input points, use
    d_out >= 2*n*d_in+1 for multisets and d_out >= 2n(d_in+1)+1 for distributions.
    
    The input point tensor X should be of size (<batch_dims>, n, d_in), where <batch_dims> can be
    any list of dimensions. The accompanying weights W should be of size (<batch_dims>, n).
    The output is of size (<batch_dims>, d_out).

    If W is not provided, all weights are assumed to be uniform 1/n, with n being the number
    of points in X.
            
    The weights should be non-negative. They do not have to be normalized, as they are normalized
    internally, but they should have a positive sum. 

    Graph mode
    ----------
    If graph_mode=True, W is treated as a conjugacy matrix of a graph with vertex features given in X.
    If X is of size (<batch_dims>, n, d_in), then W should be of size (<batch_dims>, nRecipients, n),
    with W[<batch_indices>, i, j] containing the weight of the edge from vertex j to vertex i. 
    The output is then of size (<batch_dims>, nRecipients, d_out), with each vector output[<batch indices>, i, :]
    holding the embedding of all feature vectors of neighbors of vertex i, with the corresponding weights being
    the weights of the edges leading from them to i.

    Note that W does not have to be square; hence the number of message recipients needs not be equal to the number
    of senders.

    Cartesian mode
    --------------
    If d_out=None and instead nSlices and nFreqs are provided, the embedding is computed with a Cartesian product of the
    slices and frequencies. The output shape is then (<batch_dims>, nSlices, nFreqs), or in graph mode
    (<batch_dims>, nRecipients, nSlices, nFreqs).
    
    If collapse_freqs=True, then the frequency axis is collaped to the slice axis, resulting in output size
    (<batch_dims>, nSlices x nFreqs), or in graph mode (<batch_dims>, nRecipients, nSlices x nFreqs).

    Sparse W
    --------
    The input W can be sparse. In some use cases this could lead to a considerable reduction in running time and memory
    complexity. The most common use scenario is in graph mode, when W represents the adjacency matrix of a graph with
    a large number of vertices and a relatively low number of edges.
'''

class FSW_embedding(nn.Module):
    
    @type_enforced.Enforcer(enabled=True)
    def __init__(self, 
                 d_in : int, # ambient dimension of input multisets / measures
                 d_out : int | None = None, # embedding output dimension
                 nSlices : int | None = None, nFreqs : int | None = None, collapse_freqs : bool = False,
                 d_edge : int = 0, # dimension of edge feature vectors. requires calling forward() with graph_mode=True
                 encode_total_mass : bool = False, # Tells whether to encode the input multiset sizes (a.k.a. total mass) in the embedding
                 total_mass_encoding_function : str = 'identity', # 'identity' : f(x)=x, 'sqrt' : f(x) = sqrt(1+x)-1, 'log' : log(1+x)
                 total_mass_encoding_scale : int | float = 1.0,                 
                 total_mass_encoding_method : str = 'plain', # 'plain' / 'homog' / 'homog_alt'
                 total_mass_pad_thresh : float | int = 1.0, # Multisets/mesaures with a total mass below that threshold get padded with the complementary mass placed at x=0
                 learnable_slices : bool = False, learnable_freqs : bool = False, learnable_total_mass_encoding_scale : bool = False,
                 freqs_init : float | int | str | tuple[float,float] = 'random',
                 minimize_slice_coherence : bool = False,
                 enable_bias : bool = True,
                 device : torch.device | int | str | None = None, dtype : torch.dtype = torch.float32, 
                 load_custom_cuda_lib = True,
                 report : bool = False, user_warnings : bool = True,
                 report_on_coherence_minimization : bool = False):

        super().__init__()

        self.user_warnings = user_warnings

        # Load custom CUDA library
        global libfsw_embedding
        if load_custom_cuda_lib and (libfsw_embedding is None):            
            try:
                libfsw_embedding = ctypes.CDLL(libfsw_embedding_path)
                qprintln(report, 'Loaded custom CUDA library \'%s\'' % (libfsw_embedding_path))
            except OSError as e:
                if fsw_embedding_produce_error_on_custom_library_loading_failure:
                    raise RuntimeError('Error loading custom CUDA library \'%s\'. Try rebuilding it by running the script file \'build_fsw_embedding\' found at the same directory. If this doesn\'t help, this error can be circumvented by setting load_custom_cuda_lib=False at FSW_embedding() init, or setting fsw_embedding_produce_error_on_custom_library_loading_failure=False at FSW_embedding.py code. This will use pure PyTorch code instead of the custom CUDA library, which is a bit slower.' % (libfsw_embedding_path))
                elif self.user_warnings:
                    warnings.warn('Could not load custom CUDA library \'%s\'. Using pure torch implementation, which is a bit slower. (To silence, set user_warnings=False at FSW_embedding() init)' % (libfsw_embedding_path), UserWarning)
                libfsw_embedding = None

        # Process sizes
        assert d_in >= 0, 'd_in must be nonnegative'
        assert d_edge >= 0, 'd_edge must be nonnegative'
        assert (d_out is None) or (d_out >= 0), 'd_out must be nonnegative or None'

        if d_out == 0:
            # If the output should be empty, we force encode_total_mass to be False
            encode_total_mass = 0

        self.d_in = d_in
        self.d_edge = d_edge

        self.encode_total_mass = encode_total_mass
        self.total_mass_encoding_dim = 1 if self.encode_total_mass else 0
        self.total_mass_encoding_scale_init = total_mass_encoding_scale

        total_mass_pad_thresh = float(total_mass_pad_thresh)
        assert not np.isinf(total_mass_pad_thresh), 'total_mass_pad_thresh cannot be inf'
        assert not np.isnan(total_mass_pad_thresh), 'total_mass_pad_thresh cannot be NaN'        
        assert total_mass_pad_thresh > 0, 'total_mass_pad_thresh must be positive'

        self.total_mass_pad_thresh = total_mass_pad_thresh

        assert total_mass_encoding_method in {'plain','homog','homog_alt'}, '<total_mass_encoding_method> must be one of \'plain\', \'homog\', \'homog_alg\''
        self.total_mass_encoding_method = total_mass_encoding_method

        assert total_mass_encoding_function in {'identity','sqrt','log'}, '<total_mass_encoding_function> must be one of \'identity\', \'sqrt\', \'log\''
        self.total_mass_encoding_function = total_mass_encoding_function        

        if self.d_edge == 0:
            input_space_name = 'R^%d' % (self.d_in)
        else:
            input_space_name = 'R^(%d+%d)' % (self.d_in, self.d_edge)

        if (d_out is not None) and (nSlices is None) and (nFreqs is None):
            self.cartesian_mode = False            
            self.collapse_freqs = False
            self.d_out = d_out
            self.nSlices = d_out - self.total_mass_encoding_dim
            self.nFreqs = d_out - self.total_mass_encoding_dim
            output_space_name = 'R^%d' % (self.d_out)

        elif (d_out is None) and (nSlices is not None) and (nFreqs is not None):
            assert collapse_freqs or (not encode_total_mass), 'Cartesian mode with collapse_freqs=False is not supported when encode_total_mass=True'

            self.cartesian_mode = True
            self.collapse_freqs = collapse_freqs
            self.nSlices = nSlices
            self.nFreqs = nFreqs
            self.d_out = nSlices * nFreqs + self.total_mass_encoding_dim
            output_space_name = ('R^%d' % (self.d_out)) if self.collapse_freqs else ('R^(%d\u00d7%d)' % (self.nSlices, self.nFreqs))

        else:
            assert False, "Expected exactly one of (d_out != None) or (nSlices != None and nFreqs != None)"

        assert self.d_out >= 0, 'd_out must be nonnegative'

        d_out = self.d_out
        nSlices = self.nSlices
        nFreqs = self.nFreqs

        self.minimize_slice_coherence = minimize_slice_coherence

        self.learnable_slices = learnable_slices
        self.learnable_freqs = learnable_freqs
        self.learnable_total_mass_encoding_scale = learnable_total_mass_encoding_scale

        # Note: freqs_init is checked for correctness downstream at generate_embedding_parameters()
        self.freqs_init = freqs_init

        self.enable_bias = enable_bias

        # device_new and dtype_new are only defined here on __init__ and passed on to reset_parameters(), which then deletes them
        self.device_new = ifnone(device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        assert dtype.is_floating_point and (not dtype.is_complex), 'dtype must be real floating-point; instead got dtype=%s' % (dtype)
        self.dtype_new = dtype

        self.report = report
        self.report_on_coherence_minimization = report_on_coherence_minimization

        qprintln(report)
        qprintln(report, 'Fourier Sliced-Wasserstein Embedding')
        qprintln(report, 'version %s, %s' % (version, version_date))

        qprintln(report)
        qprintln(report, 'Tal Amir, Nadav Dym')
        qprintln(report, 'Technion Institute of Technology, Haifa, Israel')

        qprintln(report)
        qprintln(report, 'Based on our paper titled "Fourier Sliced-Wasserstrin Embedding for Multisets and Measures", 2024')

        qprintln(report)
        qprintln(report, 'Constructing embedding for sets in %s into %s  ' % (input_space_name, output_space_name))

        if self.cartesian_mode and self.collapse_freqs:
            slice_freq_str = 'Using %d slices \u00d7 %d frequencies, collapsed to one %d dimensional axis; ' % (nSlices, nFreqs, nSlices*nFreqs)
        elif self.cartesian_mode:
            slice_freq_str = 'Using %d slices \u00d7 %s frequencies; ' % (nSlices, nFreqs)
        else:
            slice_freq_str = 'Using %d (slice, frequency) pairs; ' % (nSlices)

        qprint(report, slice_freq_str)

        if self.learnable_slices and self.learnable_freqs:
            if self.enable_bias:
                learnable_str = 'learnable slices, frequences and biases'
            else:
                learnable_str = 'learnable slices and frequences, no bias'
        elif self.learnable_slices:
            if self.enable_bias:
                learnable_str = 'learnable slices and biases, fixed frequencies'
            else:
                learnable_str = 'learnable slices, fixed frequences, no bias'
        elif self.learnable_freqs:
            if self.enable_bias:
                learnable_str = 'fixed slices, learnable frequencies, fixed biases (initialized to zero)'
            else:
                learnable_str = 'fixed slices, learnable frequencies, no biases'
        else:
            if self.enable_bias:
                learnable_str = 'fixed slices and frequencies, fixed biases (initialized to zero)'
            else:
                learnable_str = 'fixed slices and frequencies, no bias'

        qprintln(report, learnable_str)

        qprintln(report, 'device: %s    dtype: %s' % (self.device_new, self.dtype_new))

        self.reset_parameters()



    # Resets the model parameters (projection vectors and frequencies) and updates the model settings.
    @type_enforced.Enforcer(enabled=True)
    def reset_parameters(self,
                         freqs_init : float | int | str | None | tuple[float,float] = None,
                         minimize_slice_coherence : bool | None = None,
                         report : bool | None = None,
                         report_on_coherence_minimization : bool | None = None):

        # Apply user updates for these parameters
        self.freqs_init = ifnone(freqs_init, self.freqs_init)
        self.minimize_slice_coherence = ifnone(minimize_slice_coherence, self.minimize_slice_coherence)
        self.report = ifnone(report, self.report)
        self.report_on_coherence_minimization = ifnone(report_on_coherence_minimization, self.report_on_coherence_minimization)

        # To make sure we don't use these values inside the function; if any of then is None, we must use its self. counterpart.
        del minimize_slice_coherence, freqs_init, report, report_on_coherence_minimization

        qprintln(self.report)

        if hasattr(self, 'device_new'):
            qprintln(self.report, 'Generating embedding parameters:')
        else:
            qprintln(self.report, 'Resetting embedding parameters:')


        # If we're running for the first time, get the device and dtype that were set in the __init__ method;
        # otherwise use the current device and dtype.
        if hasattr(self, 'device_new'):
            device = self.device_new
            delattr(self, 'device_new')
        else:
            device = self.get_device()

        if hasattr(self, 'dtype_new'):
            dtype = self.dtype_new
            delattr(self, 'dtype_new')
        else:
            dtype = self.get_dtype()

        # Generate projection vectors and frequencies
        # We always generate (and optimize) them in float64 and then convert to the desired dtype.
        projVecs, freqs, bias, total_mass_encoding_scale = FSW_embedding.generate_embedding_parameters(d_in=self.d_in+self.d_edge, 
                                                                           nSlices=self.nSlices, nFreqs=self.nFreqs,
                                                                           cartesian_mode=self.cartesian_mode,
                                                                           collapse_freqs=self.collapse_freqs,
                                                                           total_mass_encoding_dim=self.total_mass_encoding_dim,
                                                                           total_mass_encoding_scale_init=self.total_mass_encoding_scale_init,
                                                                           freqs_init=self.freqs_init,
                                                                           minimize_slice_coherence=self.minimize_slice_coherence,
                                                                           device=device,
                                                                           report = self.report,
                                                                           report_on_coherence_minimization = self.report_on_coherence_minimization)

        projVecs = projVecs.to(dtype=dtype, device=device)
        freqs = freqs.to(dtype=dtype, device=device)

        self.projVecs = nn.Parameter( projVecs, requires_grad=self.learnable_slices )
        self.freqs = nn.Parameter( freqs, requires_grad=self.learnable_freqs )

        if self.enable_bias:
            bias = bias.to(dtype=dtype, device=device)

            if self.cartesian_mode and self.collapse_freqs:
                bias = bias.reshape((self.nSlices*self.nFreqs))

            self.bias = nn.Parameter( bias, requires_grad=self.learnable_slices )

        if self.encode_total_mass:
            self.total_mass_encoding_scale = nn.Parameter( total_mass_encoding_scale, requires_grad=self.learnable_total_mass_encoding_scale )

        # This also initializes the .device and .dtype fields
        self.to(device=self.get_device(), dtype=self.get_dtype())

        return self



    def to(self, *args, **kwargs):
        if 'dtype' in kwargs:
            arg = kwargs['dtype']

            assert isinstance(arg, torch.dtype), 'invalid input type %s at argument ''dtype''' % (type(arg))
            assert arg.is_floating_point and not arg.is_complex, 'dtype must be real floating-point; instead got dtype=%s' % (arg)

        for arg in args:
            if isinstance(arg, torch.dtype):
                assert arg.is_floating_point and not arg.is_complex, 'dtype must be real floating-point; instead got dtype=%s' % (arg)

        super().to(*args, **kwargs)       
       
        return self



    def get_device(self):
        return self.projVecs.device



    def get_dtype(self):
        return self.projVecs.dtype
    


    def generate_embedding_parameters(d_in, nSlices, nFreqs, cartesian_mode, collapse_freqs,
                                      total_mass_encoding_dim, total_mass_encoding_scale_init,
                                      freqs_init, minimize_slice_coherence, device, report, report_on_coherence_minimization):
        dtype_init = torch.float64

        # Axis number for the ambient space R^d_in
        ambspace_axis = 1

        ### A. Generate projection vectors

        projVecs = torch.randn(size=(nSlices, d_in), dtype=dtype_init, device=device)
        projVecs = nn.functional.normalize(projVecs, p=2.0, dim=ambspace_axis, eps=0, out=None)

        if minimize_slice_coherence:
            # We use mutual-coherence minimization even when nSlices <= d_in, where we could use
            # the QR or SVD methods below, since in low-memory conditions these methods sometimes
            # crash out of memory, whereas minimize_mutual_coherence() works well.
            #TODO: Deal with degenerate cases such as d_in = 1 or nSlices=1, nSlices=0
            if (nSlices > d_in) or True:
                projVecs = minimize_mutual_coherence(projVecs, report=report_on_coherence_minimization)
                qprintln(report, '- Generated %d projection vectors in R^%d with minimized mutual coherence' % (nSlices, d_in))
            else:
                # Here we need to compute a set of nSlices orthogonal vectors in R^d_in.
                # Below are two methods to do so: SVD and QR decomposition.
                if True:
                    U, S, Vh = torch.linalg.svd(projVecs, full_matrices=False)
                    projVecs = Vh
                    del U, S, Vh
                else:                    
                    projVecs = projVecs.transpose(0,1)
                    projVecs, R = torch.linalg.qr(projVecs, mode='reduced')
                    del R
                    projVecs = projVecs.transpose(0,1)
                qprintln(report, '- Generated %d perpendicular projection vectors in R^%d using QR decomposition' % (nSlices, d_in))
        else:
            qprintln(report, '- Generated %d random projection vectors' % (nSlices))

        # Detect nans, infs and zero vectors in projVecs
        assert not torch.isinf(projVecs).any(), "Found infs in projVecs"
        assert not torch.isnan(projVecs).any(), "Found nans in projVecs"
        assert not (projVecs == 0).all(dim=1).any(), 'Found zero vectors in projVecs'



        ### B. Generate frequencies
        freqs_shape = (nFreqs,) # Note: Changing this to (self.nFreqs, 1) yields incorrect results in self.forward()

        if nFreqs == 0:
            freqs = torch.zeros(size=freqs_shape, dtype=dtype_init, device=device)
            qprintln(report, '- Initialized 0 frequencies')

        elif isinstance(freqs_init, numbers.Real):
            assert not np.isinf(freqs_init), 'freqs_init cannot be infinite'
            assert not np.isnan(freqs_init), 'freqs_init cannot be NaN'
            freqs = freqs_init * torch.ones(size=freqs_shape, dtype=dtype_init, device=device)
            qprintln(report, '- Initialized %d frequencies to %g' % (nFreqs, freqs_init))

        elif isinstance(freqs_init, tuple):            
            # Here freqs_init should have been type-enforced to be a tuple of two real numbers.
            # However, it does not prevent the tuple from containing more numbers.
            assert(len(freqs_init)==2), 'When freqs_init is a tuple, it must be of length 2'

            a = freqs_init[0]
            b = freqs_init[1]
            assert not np.isinf(a) and not np.isinf(b), 'Received infinite value in freqs_init tuple'
            assert not np.isnan(a) and not np.isnan(b), 'Received NaN value in freqs_init tuple'
            assert a <= b, 'When freqs_init is a tuple, it is required to satisfy freqs_init[0] <= freqs_init[1]'

            if nFreqs == 1:
                freqs = a + (b-a)/2 * torch.ones(size=freqs_shape, dtype=dtype_init, device=device)
            else:
                freqs = a + (b-a) * ( torch.arange(nFreqs, dtype=dtype_init, device=device) / (nFreqs-1) )

            qprintln(report, '- Initialized %d equispaced frequencies in the interval [%d, %d]' % (nFreqs,a,b))

        elif freqs_init == 'random':
            freqs = torch.rand(size=freqs_shape, dtype=dtype_init, device=device)
            freqs, junk = torch.sort(freqs, dim=0)
            assert (freqs != 1).all(), "Unexpected behavior of torch.rand(): Returned a value of 1, whereas values are supposed to be in [0,1)"
            assert (freqs < 1).all(), "Unexpected behavior of torch.rand(): Returned a value > 1, whereas values are supposed to be in [0,1)"
            freqs = freqs / (1-freqs)

            qprintln(report, '- Initialized %d random frequencies i.i.d. with density f(x) = 1/(1+x)^2, x\u22650' % (nFreqs))

        elif freqs_init == 'spread':
            freqs = ( 0.5 + torch.arange(nFreqs, dtype=dtype_init, device=device).reshape(freqs_shape) ) / nFreqs
            freqs = freqs / (1-freqs)
            qprintln(report, '- Initialized %d frequencies spread evenly in [%g, %g] according to probability density' % (nFreqs, freqs[0], freqs[-1]))

        else:
            raise RuntimeError('Invalid value for argument freqs_init; expected number, tuple (a,b) of numbers denoting an interval, \'random\' or \'spread\'')
        
        # Detect nan and inf entries in freqs
        if nFreqs > 0:
            assert not torch.isinf(freqs).any(), "Found infs in freqs"
            assert not torch.isnan(freqs).any(), "Found nans in freqs"

        # C. Generate bias vector. Always initialized to zero.
        if cartesian_mode and not collapse_freqs:
            bias_shape = (nSlices, nFreqs)
        elif cartesian_mode and collapse_freqs:
            bias_shape = (nSlices*nFreqs + total_mass_encoding_dim,)
        else:
            bias_shape = (nSlices + total_mass_encoding_dim,)

        bias = torch.zeros(size=bias_shape, dtype=dtype_init, device=device)

        if total_mass_encoding_dim > 0:
            total_mass_encoding_scale = torch.tensor(total_mass_encoding_scale_init, device=device, dtype=dtype_init)
        else:
            total_mass_encoding_scale = None

        qprintln(report)

        return projVecs, freqs, bias, total_mass_encoding_scale



    # Spreads the frequencies on an interval centered at 'center' with the given radius, in an equispaced manner.
    # This might be useful when using the embedding for graph message passing with learnable_projs=True, as the magnitude of the
    # projection vectors already determines the effective frequency, and having a very high max-frequency-to-low-frequency ratio
    # may impede the optimization due to ill conditioning.
    @type_enforced.Enforcer(enabled=True)
    def spread_freqs_at_interval(self, center : float | int, radius : float | int):
        assert radius >= 0

        if (self.nFreqs == 1) or (radius == 0):
            freqs_new = center * torch.ones_like(self.freqs)
        else:
            spread = 2 * (0.5 + torch.arange(self.nFreqs, dtype=self.get_dtype(), device=self.get_device()).reshape(self.freqs.shape) ) / self.nFreqs - 1
            spread = spread * 1/(1 - 1/self.nFreqs)
            freqs_new = center + radius * spread

        state_dict = self.state_dict()
        state_dict['freqs'] = freqs_new
        self.load_state_dict(state_dict)

        return self


    # Forward: 8 seconds, backward: 1.73 seconds
    @type_enforced.Enforcer(enabled=True)
    def forward(self, X : torch.Tensor, W : torch.Tensor | str = 'unit', X_edge : torch.Tensor | None = None, 
                graph_mode : bool = False, serialize_num_slices : int | None = None):
        # Simple use:
        # X sized (n,d_in) represents a multiset of n points in R^d_in.
        # If W sized (n,) is provided, then (X,W) represents a distribution, with each point X[i,:]
        # assigned the weight W[i]. 
        # The weights are normalized internally so they do not have to sum up to 1, but they must be nonnegative
        # and contain at least one nonzero.
        # If W is not provided, it is assumed to be uniform weights 1/n.
        # The output embedding is of size (d_out,).

        # Batches:
        # Batches of distributions as above can be provided via X of size (<batch_dims>, n, d_in) and W (<batch_dims>, n).
        # Here the output will be of size (<batch_dims>, d_out).

        # Graph mode: (Requires that W be given explicitly)
        # If graph_mode=True, then the points in X are shared between all batches along the last batch dimension.
        # That is, forward(X,W,graph_mode=True) produces (more efficiently) the same result as forward(X_expand, W),
        # where d_in = X.shape[-1]  and  X_expand = X.unsqueeze(dim=-3).expand( tuple(W.shape) + (d_in,) ).
        # A common usage for this feature is when W sized (n,n) represents an adjacency matrix of a graph, 
        # and X sized (n,d_in) represents vertex features. Then the output of embed(X,W,graph_mode=True) will be of size (n,d_out),
        # with each of its [i,:] rows representing the embedding of the features of all neighbours of vertex i, with weights
        # given by their edge weights.
        # The input in graph mode can be batched as follows: W of size (<batch_dims>, n) with X of size (batch_dims[:-1], d_in)
        # Note that batch_dims[-1] is not required to equal n, i.e. it is possible to process subblocks of adjacency matrices.

        # Output:
        # X_emb: The Sliced-Wasserstein embedding of the input distributions.
        #
        # By default the output size is (<batch_dims>, d_out), where each X_emb(j1,...,jk,:) contains the embedding of one distribution.
        #
        # If the parameters nSlices and nFreqs are set on initialization, then the output is of size
        # (<batch_dims>, nSlices, nFreqs) or (<batch_dims>, nSlices*nFreqs) - the former if 
        # the parameter 'collapse_freqs' is set to False (default), the latter if it is set to True.
        #
        # If serialize_num_slices = t (integer), then the computation is serialized to batches of size t.
        # This does not affect the result, but reduces the momery complexity by a factor of <number of slices> / t

        # Verify slices and frequencies at each forward pass if they are learnable
        if fsw_embedding_basic_safety_checks and self.learnable_slices:
            assert not torch.isnan(self.projVecs).any(), 'Projection vectors contain NaNs'
            assert not torch.isinf(self.projVecs).any(), 'Projection vectors contain infs'
            # Note: We allow them to contain zero vectors when they are learnable, in case i.e. when sparsity is desired
            # assert not (self.projVecs == 0).all(dim=1).any(), 'Projection vectors contain a zero vector'

        if fsw_embedding_basic_safety_checks and self.learnable_freqs:
            assert not torch.isnan(self.freqs).any(), 'Frequencies contain NaNs'
            assert not torch.isinf(self.freqs).any(), 'Frequencies contain infs'

        ### A. Verify input types and content

        assert self.total_mass_pad_thresh > 0, 'total_mass_pad_thresh must be positive'

        if self.d_edge > 0:
            assert graph_mode, 'd_edge > 0 (given at initialization) necessitates graph_mode=True on forward call'
            assert X_edge is not None, 'X_edge must be provided since d_edge > 0'            
        else:
            assert (X_edge is None) or (X_edge.numel() == 0), 'X_edge should be None or empty since d_edge == 0'
            X_edge = None

        assert torch.is_tensor(X), 'X must be a pytorch tensor. Instead got type %s' % (type(X))
        assert torch.is_tensor(W) or W in {'unit', 'uniform'}, 'W must be a pytorch tensor, \'unit\' or \'uniform\''
        assert X.dtype == self.get_dtype(), ( "X has the wrong dtype. Expected %s, got %s" % (self.get_dtype(), X.dtype) )
        assert X.device == self.get_device(), ( "X is on the wrong device. Expected %s, got %s" % (self.get_device(), X.device) )

        if fsw_embedding_basic_safety_checks:
            assert not torch.isnan(X).any(), "The entries of X cannot contain NaNs"
            assert not torch.isinf(X).any(), "All entries of X must be finite"

        if torch.is_tensor(W):
            assert W.dtype == self.get_dtype(), ( "W has the wrong dtype. Expected %s, got %s" % (self.get_dtype(), W.dtype) )
            assert W.device == self.get_device(), ( "W is on the wrong device. Expected %s, got %s" % (self.get_device(), W.device) )

            # Check if W is sparse. If so, ensure that W is of the correct layout.            
            # Note: Strangely enough, sparse tensors of layouts other than COO (e.g. CSR) may have is_sparse=False.
            #       This may lead us to mistakenly treat a, e.g. W that is sparse CSR as dense.
            #       Currently there is no is_dense() function in torch, so reading the layout string directly is the second best.
            if W.is_sparse or W.layout != torch.strided:
                assert W.layout == torch.sparse_coo, ( "Sparse W has an unsupported sparsity layout '%s'. Only the COO layout (torch.sparse_coo) is currently supported." % (W.layout) )

                assert W.is_coalesced(), 'Sparse W must be coalesced'
                assert W.dense_dim() == 0, 'W.dense_dim() must be zero'

                if fsw_embedding_basic_safety_checks:
                    W_vals = W.values()
            else:
                if fsw_embedding_basic_safety_checks:
                    W_vals = W

            if fsw_embedding_basic_safety_checks:
                assert not torch.isnan(W_vals).any(), "W cannot contain NaNs"
                assert not torch.isinf(W_vals).any(), "All entries of W must be finite"
                assert (W_vals >= 0).all(), "All entries of W must be nonnegative"
                del W_vals

        if X_edge is not None:
            assert torch.is_tensor(W), 'When X_edge is provided, W must be provided explicitly'
            assert (X_edge.device == self.get_device()), ( "X_edge is on the wrong device. Expected %s, got %s" % (self.get_device(), X_edge.device) )
            assert (X_edge.dtype == self.get_dtype()), ( "X_edge has the wrong dtype. Expected %s, got %s" % (self.get_dtype(), X_edge.dtype) )        

            if X_edge.is_sparse or X_edge.layout != torch.strided:
                assert X_edge.layout == torch.sparse_coo, ( "Sparse X_edge has an unsupported sparsity layout '%s'. Only the COO layout (torch.sparse_coo) is currently supported." % (X_edge.layout) )

                assert X_edge.is_coalesced(), 'Sparse X_edge must be coalesced'
                assert X_edge.dense_dim() in (0,1), 'X_edge.dense_dim() must be 1 or 0'
                assert (self.d_edge == 1) or (X_edge.dense_dim() == 1), 'X_edge.dense_dim() must be 1 since d_edge > 1'

                if fsw_embedding_basic_safety_checks:
                    X_edge_vals = X_edge.values()
            else:
                if fsw_embedding_basic_safety_checks:
                    X_edge_vals = X_edge

            if fsw_embedding_basic_safety_checks:
                assert not torch.isnan(X_edge_vals).any(), "X_edge_vals cannot contain NaNs"
                assert not torch.isinf(X_edge_vals).any(), "All entries of X_edge_vals must be finite"
                del X_edge_vals

            assert X_edge.is_sparse == W.is_sparse, 'X_edge and W must either both or neither be sparse'


        ### B. Verify input sizes
            
        assert len(X.shape) >= 2, "X must be a tensor of order at least 2"
        assert X.shape[-1] == self.d_in, "The last dimension of X must equal d_in=%d. Instead got %d" % (self.d_in, X.shape[-1])        

        if not graph_mode:
            # batch_dims contains everything that precedes (n,d_in) in X.shape
            batch_dims = tuple(X.shape[0:-2])
            n = X.shape[len(batch_dims)]

            if torch.is_tensor(W):
                if (len(W.shape) == len(X.shape)) and (W.shape[-1] == X.shape[-2]) and (W.shape[0:-2] == X.shape[0:-2]):
                    err_str = "Shape mismatch between X and W: If X.shape = (b1,b2,...,bk,n,d_in) then W.shape should be (b1,b2,...,bk,n) (Perhaps missing argument graph_mode=True?)"
                else:
                    err_str = "Shape mismatch between X and W: If X.shape = (b1,b2,...,bk,n,d_in) then W.shape should be (b1,b2,...,bk,n) (unless graph_mode=True)"
                
                assert (len(W.shape) == len(X.shape)-1) and (W.shape == X.shape[0:-1]), err_str

            elif W == 'unit':
                # Initialize with unit weights
                W = torch.ones(batch_dims + (n,), dtype=self.get_dtype(), device=self.get_device())

            elif W == 'uniform':
                # Initialize with uniform weights
                W = torch.full(batch_dims + (n,), 1.0/n, dtype=self.get_dtype(), device=self.get_device())

        elif graph_mode:
            assert torch.is_tensor(W), 'W must be explicitly provided when graph_mode=True'

            # batch_dims contains everything that precedes (nRecipients, n) in W.shape
            batch_dims = tuple(W.shape[0:-2])
            nRecipients = W.shape[-2]
            n = W.shape[-1]

            assert (len(W.shape) == len(X.shape)) and (W.shape[-1] == X.shape[-2]) and (W.shape[0:-2] == X.shape[0:-2]), "Shape mismatch between X and W: When graph_mode=True, if W.shape = (b1,b2,...,bk,nRecipients,n) then X.shape should be (b1,b2,...,bk,n,d_in)"

            if X_edge is not None:
                # Verify that X_edge has the right shape and is compatible with W
                assert (((self.d_edge == 1) and (X_edge.shape == W.shape)) or 
                    ((X_edge.dim() == W.dim()+1) and (X_edge.shape[0:-1] == W.shape) and (X_edge.shape[-1] == self.d_edge))), (
                    "Shape mismatch between X_edge and W: if W.shape = (b1,b2,...,bk,nRecipients,n) then X.shape should be (b1,b2,...,bk,nRecipients,n,d_edge) (with the possible exception (b1,b2,...,bk,nRecipients,n) when d_edge=1" )
                
                if X_edge.is_sparse:
                    assert X_edge.values().shape[0] == W.values().shape[0], 'Sparse X_edge must have the same number of values() as W'
                    if fsw_embedding_basic_safety_checks:
                        assert (X_edge.indices() == W.indices()).all(), 'Sparse X_edge must have the same nonzero pattern as W'

                    if X_edge.dense_dim()==0:
                        X_edge = ag.unsqueeze_dense_dim.apply(X_edge)
                    

        ### C. Precalculate axis indices and output shape
             
        # These are the different axes we use to store data for processing. These definitions are repeated in forward_helper()
        # element_axis corresponds to the index of the multiset elements
        # ambspace_axis corresponds to the elements' coordinate index in the ambient space R^d_in
        # After projection, the ambient space coordinates are replaced by the projection/slice number; thus proj_axis=ambspace_axis
        # If we're in Cartesian mode, the frequencies have their own axis freq_axis, otherwise it is the same axis as proj_axis.
        recipient_axis = len(batch_dims) if graph_mode else None  # Message-recipient vertices
        element_axis  = recipient_axis+1 if graph_mode else len(batch_dims) # In graph mode this axis denotes the message-sender vertices
        ambspace_axis = element_axis + 1        
        proj_axis     = ambspace_axis
        freq_axis     = proj_axis +1 if self.cartesian_mode else proj_axis
        output_proj_axis = element_axis # In the output, the element axis is replaced by the projection axis

        output_shape_before_collapse_and_totmass_augmentation =  batch_dims + (nRecipients,) if graph_mode else batch_dims
        output_shape_before_collapse_and_totmass_augmentation += (self.nSlices, self.nFreqs) if self.cartesian_mode else (self.nSlices, )

        ### D. Input is ok. Start working.

        # Calculate W_sum, which contains the total mass of the input measures
        if W.is_sparse:
            slice_info_W = sp.get_slice_info(W, -1)
            W_sum = ag.sum_sparseToDense.apply(W, -1, slice_info_W)
        
        else:
            W_sum = torch.sum(W, dim=-1, keepdim=True)            

        # Total-mass deficit to be compensated for by padding
        W_pad = ag.custom_lowclamp.apply(self.total_mass_pad_thresh-W_sum, 0.0)

        # Detect weight deficit and augment W and X accordingly
        if (W_pad > 0).any():
            zshape = list(X.shape)
            zshape[-2] = 1
            X = torch.cat( (X, torch.zeros(zshape, device=X.device, dtype=X.dtype)), dim=-2 )               

            if W.is_sparse:
                # Make sure this works
                W_pad = sp.to_sparse_full(W_pad)
                W = ag.concat_sparse.apply(W, W_pad)
                slice_info_W = sp.get_slice_info(W, -1)

                if X_edge is not None:
                    X_edge_pad_inds = W_pad.indices()
                    X_edge_pad_vals = torch.zeros( (nRecipients, self.d_edge), device=X.device, dtype=X.dtype)
                    X_edge_pad_shape = replace_in_tuple(tuple(X_edge.shape), -2, 1)

                    X_edge_pad = sp.sparse_coo_tensor_coalesced(indices=X_edge_pad_inds, values=X_edge_pad_vals, size = X_edge_pad_shape)
                    X_edge = ag.concat_sparse.apply(X_edge, X_edge_pad)
            else:
                W = torch.cat( (W, W_pad), dim=-1 )
                if X_edge is not None:
                    zshape = list(W.shape)+[self.d_edge,]
                    zshape[-2] = 1
                    if X_edge.dim() == W.dim():
                        X_edge = X_edge.unsqueeze(-1)
                    X_edge = torch.cat( (X_edge, torch.zeros(zshape, device=X.device, dtype=X.dtype)), dim=-2 )

            W_sum_padded = ag.custom_lowclamp.apply(W_sum, self.total_mass_pad_thresh)
        else:            
            W_sum_padded = W_sum
        
        del W_pad

        # Normalize W according to W_sum_padded
        if W.is_sparse:
            W = ag.div_sparse_dense.apply(W, W_sum_padded, slice_info_W)          
            del slice_info_W, W_sum_padded
        else:
            W = W / W_sum_padded
            del W_sum_padded

        # For compatibility reasons, we support the case of zero-dimensional output tensor
        if self.d_out == 0:
            X_emb = torch.zeros(size=output_shape_before_collapse_and_totmass_augmentation, dtype=self.get_dtype(), device=self.get_device())

        elif (serialize_num_slices is None) or (serialize_num_slices >= self.nSlices):
            X_emb = FSW_embedding.forward_helper(X, W, self.projVecs, self.freqs, graph_mode, X_edge, self.cartesian_mode, batch_dims)

        else:
            assert isinstance(serialize_num_slices, int) and (serialize_num_slices >= 1), 'serialize_num_slices must be None or a positive integer'

            nIter = (self.nSlices // serialize_num_slices) if (self.nSlices % serialize_num_slices == 0) else (1 + self.nSlices // serialize_num_slices)

            X_emb = torch.empty(size=output_shape_before_collapse_and_totmass_augmentation, dtype=self.get_dtype(), device=self.get_device())

            for iIter in range(nIter):
                inds_curr = torch.arange(iIter*serialize_num_slices, min( self.nSlices, (iIter+1)*serialize_num_slices), dtype=torch.int64, device=self.get_device())
                projVecs_curr = self.projVecs[inds_curr,:]
                freqs_curr = self.freqs if self.cartesian_mode else self.freqs[inds_curr]
                
                out_curr = FSW_embedding.forward_helper(X, W, projVecs_curr, freqs_curr, graph_mode, X_edge, self.cartesian_mode, batch_dims)                
                assign_at(X_emb, out_curr, output_proj_axis, inds_curr)

        if self.cartesian_mode and self.collapse_freqs:
            X_emb = torch.flatten(X_emb, start_dim=element_axis, end_dim=element_axis+1)

        if self.encode_total_mass:
            if self.total_mass_encoding_function == 'identity':
                total_mass = W_sum
            elif self.total_mass_encoding_function == 'sqrt':
                # x/(sqrt(x+1)+1) is a numerically-safe formulation of sqrt(1+x)-1
                # note that we don't use sqrt(1+x) since we need the function to vanish at x=0,
                # and we don't use sqrt(x) since we need it to have a gradient at x=0.
                total_mass = 2*( W_sum / ( torch.sqrt(W_sum + 1) + 1) )
            elif self.total_mass_encoding_function == 'log':
                total_mass = torch.log1p(W_sum)
            else:
                raise RuntimeError('This should not happen')
            
            del W_sum
            
            #print('Total mass scale: ', self.total_mass_encoding_scale.item(), 'requires grad: ', self.total_mass_encoding_scale.requires_grad)
            total_mass *= self.total_mass_encoding_scale

            if self.total_mass_encoding_method == 'plain':
                X_emb = torch.cat( (total_mass, X_emb), dim=-1)
            elif self.total_mass_encoding_method == 'homog':
                X_emb_norm = torch.mean(X_emb.abs(), dim=-1, keepdim=True)
                X_emb = torch.cat( (total_mass * X_emb_norm, X_emb), dim=-1)
            elif self.total_mass_encoding_method == 'homog_alt':
                X_emb_norm = torch.mean(X_emb.abs(), dim=-1, keepdim=True)
                X_emb = torch.cat( (FSW_embedding.total_mass_homog_alt_encoding_part1(total_mass) * X_emb_norm,
                                    FSW_embedding.total_mass_homog_alt_encoding_part2(total_mass) * X_emb), dim=-1)            
            else:
                raise RuntimeError('This should not happen')

        # Add bias
        if self.enable_bias:
            X_emb += self.bias

        return X_emb


    # 9.7 seconds
    def forward_helper(X, W, projVecs, freqs, graph_mode, X_edge, cartesian_mode, batch_dims):
        # This function computes the embedding of (X,W) for a subset of the projections and frequencies.
        # projVecs should be of size (num_projections x d_in), and freqs should be of size nFreqs (not nFreqs x 1).

        d_in = X.shape[-1]
        n = W.shape[-1]
        nRecepients = W.shape[-2] if graph_mode else None
        nProjs = projVecs.shape[0]
        nFreqs = len(freqs)
        sparse_mode = W.is_sparse
        d_edge = X_edge.shape[-1] if X_edge is not None else 0

        assert len(freqs.shape) == 1, "This should not happen"
        assert (len(projVecs.shape) == 2) and (projVecs.shape[1] == d_in + d_edge), "This should not happen"

        # Calculate the projections of X 
        if d_edge == 0:
            Xp = torch.tensordot(X, projVecs, dims=((-1,),(1,)))
        else:
            Xp = torch.tensordot(X, projVecs[:,0:d_in], dims=((-1,),(1,)))

        del X

        if d_edge == 0:
            # Sort the projected elements 
            # Note: We sort before the graph-mode expansion because it makes things simpler in the case when W is sparse

            # Sort along element/sender axis
            if sparse_mode:
                Xps, Xpi = ag.sort.apply(Xp, -2, False)
            else:
                Xps, Xpi = torch.sort(Xp, dim=-2, descending=False)

            del Xp

            if graph_mode:
                # Create recepient axis before sender axis and projection axis
                Xps = Xps.unsqueeze(dim=-3)
                Xpi = Xpi.unsqueeze(dim=-3)

        elif sparse_mode: # d_edge > 0, sparse_mode=True
            Xe = X_edge.values()
            Xep = torch.tensordot(Xe, projVecs[:,d_in:], dims=((-1,),(-1,)))
            del Xe

            inds = W.indices()

            # Remove recepient axis from inds
            dims_without_recipient = [i for i in range(inds.shape[0]) if i != len(batch_dims)]

            # For each edge, get the corresponding sender vertex feature vector after projection
            Xp_temp = Xp[tuple(inds[dims_without_recipient,:])]

            Xep += Xp_temp
            del Xp_temp, inds

            Xep_shape = replace_in_tuple(tuple(X_edge.shape), -1, Xep.shape[1])

            Xep = sp.sparse_coo_tensor_coalesced(indices=W.indices(), values=Xep, size=Xep_shape)
            Xep = ag.flatten_dense_dim.apply(Xep)
            Xeps, Xepi = ag.sort_sparse.apply(Xep, -2, False)
            del Xep

        else: # d_edge > 0, sparse_mode=False
            # Create recepient axis before sender axis and projection axis
            Xpx = Xp.unsqueeze(dim=-3) #.expand(tuple(W.shape) + (nProjs,))
            # Replicate Xpx along recepient axis
            Xpx = Xpx.repeat(replace_in_tuple((1,)*Xpx.dim(),-3,nRecepients))
            # Add edge-feature part of inner product to each recepient for each sender and projection
            Xpx += torch.tensordot(X_edge, projVecs[:,d_in:], dims=((-1,),(-1,)))
            # Sort along element/sender axis
            Xps, Xpi = ag.sort.apply(Xpx, -2, False)
            #Xps, Xpi = torch.sort(Xpx, dim=-2, descending=False, stable=True)

            del Xpx

        # Axis numbers as in the implementation of forward()
        # Note: These numbers are true only from here
        recipient_axis = len(batch_dims) if graph_mode else None  # Message-recipient vertices
        element_axis  = recipient_axis+1 if graph_mode else len(batch_dims) # In graph mode this axis denotes the message-sender vertices
        ambspace_axis = element_axis + 1        
        proj_axis     = ambspace_axis
        freq_axis     = proj_axis +1 if cartesian_mode else proj_axis
        output_proj_axis = element_axis # In the output, the element axis is replaced by the projection axis

        assert len(freqs.shape) == 1
        for i in range(freq_axis):
            freqs = freqs.unsqueeze(0)

        if not sparse_mode:
            if graph_mode and (d_edge == 0):
                Xps = Xps.expand(tuple(W.shape) + (nProjs,))
                Xpi = Xpi.expand(tuple(W.shape) + (nProjs,))

            # Sort the weights according to their corresponding projected elements
            W_big = W.unsqueeze(-1).expand_as(Xps)
            Wps = torch.gather(W_big, dim=element_axis, index=Xpi.to(torch.int64))

            if cartesian_mode:
                Wps = Wps.unsqueeze(dim=-1)
                Xps = Xps.unsqueeze(-1).expand(Xps.shape + (nFreqs,))

            # Once we have Wps we don't need W_big and Xpi
            del W_big, Xpi

            Wps_sum = torch.cumsum(Wps, dim=element_axis)

            # Here we assume sinc(x) = sin(pi*x)/(pi*x)
            sincs = 2 * Wps_sum * torch.sinc(2 * freqs * Wps_sum)
            sinc_diffs = diff_zeropad(sincs, dim=element_axis)
            del sincs

        elif sparse_mode:
            # We unsqueeze W to add a projection axis, in order to sort W according to each projection of X
            # Note: This repmat is unavoidable, because we sort the weights according to different permutations along proj_axis
            W_unsqueeze = ag.unsqueeze_sparse.apply(W,-1)
            del W

            # 1.71 seconds
            W_big = ag.repmat_sparse.apply(W_unsqueeze, nProjs, proj_axis)
            del W_unsqueeze

            if d_edge > 0:
                Wps = ag.permute_sparse_vals.apply(W_big, Xepi)
                del Xepi
            elif graph_mode: 
                # 1.82 seconds
                Wps = ag.permute_sparse.apply(W_big, element_axis, Xpi, recipient_axis)
                del Xpi
            else:
                Wps = ag.permute_sparse.apply(W_big, element_axis, Xpi, None)
                del Xpi

            # Once we have Wps we don't need W_big and Xpi
            del W_big

            # 2.6 seconds
            slice_info_elements = sp.get_slice_info(Wps, element_axis)
            Wps_sum = ag.cumsum_sparse.apply(Wps, element_axis, slice_info_elements)

            sp.verify_coalescence(Wps)
            sp.verify_coalescence(Wps_sum)

            if cartesian_mode:
                # Note:
                # These repmats may be avoided if ag.sinc_cos_sparse could take freqs as a separate input, and broadcast all inputs accordingly.
                # But sinc_diffs is of the same size as Wps and Wps_sum, so we could reduce the memory usage at most by 2/3, and only in cartesian mode.
                # This may not worth the effort.

                Wps = ag.repmat_sparse.apply(ag.unsqueeze_sparse.apply(Wps,-1), nFreqs, freq_axis)
                Wps_sum = ag.repmat_sparse.apply(ag.unsqueeze_sparse.apply(Wps_sum,-1), nFreqs, freq_axis)
                Xps = Xps.unsqueeze(-1).expand(Xps.shape + (nFreqs,))
               
            # Here we use the sum-to-product identity sin(2a)-sin(2b) = 2*sin(a-b)*cos(a+b)            
            # This formula probably leads to a loss of one significant digit, but it is much easier in the sparse case than using diff().
            
            # Variant 2 is more memory efficient
            variant = 2

            if variant == 1:
                arg2 = np.pi * freqs * (2*Wps_sum - Wps)
                del Wps_sum
                assert_coalesced(arg2)

            elif variant == 2:                               
                arg2 = ag.add_same_pattern.apply(Wps, Wps_sum, -1, 2)
                del Wps_sum
                # 1.22 seconds
                slice_info_freqs = sp.get_slice_info(arg2, sp.get_broadcast_dims_B_to_A(arg2, freqs))
                # 0.15 seconds
                arg2 = ag.mul_sparse_dense.apply(arg2, np.pi*freqs, slice_info_freqs)

            # 0.14 seconds
            arg1 = ag.mul_sparse_dense.apply(Wps, freqs, slice_info_freqs)

            sp.verify_coalescence(arg1)
            sp.verify_coalescence(arg2)

            # 0.53 seconds
            sinc_cos = ag.sinc_cos_sparse.apply(arg1, arg2)
            del arg1, arg2
            sinc_diffs = ag.mul_same_pattern.apply( Wps, sinc_cos, 2 ) 

            sp.verify_coalescence(sinc_cos)
            sp.verify_coalescence(sinc_diffs)

            del Wps, sinc_cos
           
        # From here we only need sinc_diffs and Xps               

        if sparse_mode:
            if d_edge == 0:
                # 1.4 seconds
                slice_info_Xps = sp.get_slice_info(sinc_diffs, sp.get_broadcast_dims_B_to_A(sinc_diffs, Xps))
                # 0.26 seconds
                products = ag.mul_sparse_dense.apply(sinc_diffs, Xps, slice_info_Xps)
                del sinc_diffs, Xps, slice_info_Xps
                sp.verify_coalescence(products)
            else:
                products = ag.mul_same_pattern.apply(sinc_diffs, Xeps, 1)
                del sinc_diffs, Xeps

            # 0.49 seconds
            product_sums = ag.sum_sparseToDense.apply(products, element_axis, slice_info_elements)
            del products, slice_info_elements
            
        else: # not sparse
            product_sums = torch.sum(sinc_diffs * Xps, dim=element_axis, keepdim=True)
            del sinc_diffs, Xps

        # We squeeze the element axis after having summed up along it
        product_sums = product_sums.squeeze(dim=element_axis)
        freqs = freqs.squeeze(dim=element_axis)

        # freqs and product_sums are always dense
        out = (1+freqs) * product_sums            
        del product_sums

        return out



    def get_mutual_coherence(self):
        gram = self.projVecs @ self.projVecs.transpose(0,1)
        inds = range(self.d_out)
        gram[inds,inds] = 0

        mu = torch.max(torch.abs(gram))
        
        welch_bound = np.sqrt( (self.d_out - self.d_in) / self.d_in / (self.d_out-1) ) 
        d3_bound = -1.0 + 2.0 * (2.0*(1.0-1.0/self.d_out)**2 - 1)**2

        if self.d_in == 3:
            lower_bound = d3_bound
        else:
            lower_bound = welch_bound

        #return mu, lower_bound
        return mu



    @type_enforced.Enforcer(enabled=True)
    def total_mass_homog_alt_encoding_part1(totmass : torch.Tensor):
        out = torch.where(totmass <= 1, totmass*(2-totmass), 1)
        return out

    @type_enforced.Enforcer(enabled=True)
    def total_mass_homog_alt_encoding_part2(totmass : torch.Tensor):
        out = torch.where(totmass <= 1, totmass.square(), 2*totmass-1)
        return out

#############################################################################################################
##                                                  Tools                                                  ##
#############################################################################################################

def timer_start():
    torch.cuda.synchronize(); 
    global tal_global_timer, tal_global_timer_start;
    tal_global_timer_start = time.time()
           


def timer_stop():
    torch.cuda.synchronize();
    global tal_global_timer, tal_global_timer_start;
    tal_global_timer += time.time()-tal_global_timer_start



def assert_coalesced(A):
    assert A.is_sparse, 'tensor must be sparse'

    debug = fsw_embedding_debug_mode
    if debug:
        assert A.is_coalesced(), 'tensor is not coalesced'


# Computes a finite difference with zero padding
def diff_zeropad(input, dim):
    pad_shape = replace_in_tuple(tuple(input.shape), index=dim, value=1)
    pad = torch.zeros(size=pad_shape, dtype=input.dtype, device=input.device)
    out = torch.diff(input, n=1, dim=dim, prepend=pad, append=None)
    return out



def replace_in_tuple(T, index, value):
    T = tuple(T)
    index = index if index >= 0 else ( index + len(T) )
    out = T[0:index] + (value,) + T[(index+1):len(T)]
    return out



@type_enforced.Enforcer(enabled=True)
def qprint(q : bool, s : str =''):
    assert type(q) == type(True)

    if q:
        print(s, end='')



@type_enforced.Enforcer(enabled=True)
def qprintln(q : bool, s : str =''):
    qprint(q, s+'\n')



# Performs something like target[:,:,...,:,inds,:,...,:] = source, where the argument 'inds' is given at dimension dim
def assign_at(target, source, dim, inds):
    scatter_inds_shape = replace_in_tuple((1,)*len(target.shape), dim, len(inds))               
    scatter_inds = inds.reshape(scatter_inds_shape).expand_as(source)
    target.scatter_(dim=dim, index=scatter_inds, src=source)
    


def ifnone(a,b):
    return a if (a is not None) else b



#############################################################################################################
##                                        Custom autograd functions                                        ##
#############################################################################################################

# These implementations were based on the example in
# https://pytorch.org/tutorials/beginner/examples_WithAutograd/two_layer_net_custom_function.html

# This code is based on Torch version 2.1.1, whose current support for gradients of sparse tensors sucks.
# Therefore I had to implement many of basic sparse-tensor operations used in this code myself.
# Should torch add autograd support for some of these operations in later versions, I will replace them adqeuately.

# Most actions here that take sparse tensor inputs require them to be coalesced, and actions
# that return sparse outputs return coalesced outputs.

# All custom autograd functions used here are defined under the 'ag' class.
class ag:
    # Ensure that the output gradient is coalesced
    class coalesce_grad(torch.autograd.Function):
        @staticmethod        
        def forward(ctx, A):            
            # TODO: Do we need to return A.clone() rather than A?
            return A

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            if grad_output.is_sparse and (not grad_output.is_coalesced()):
                grad_input = grad_output.coalesce()
            else:
                grad_input = grad_output

            return grad_input


    # Permutes each 1-dimensional slice of A along dimension dim according to the given permutation in perms.
    # Perms can be broadcast to the size of A along dimension broadcast_perms_dim.
    class permute_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim, perms, broadcast_perms_dim):
            assert A.is_coalesced(), 'A must be coalesced'
            sp.verify_coalescence(A)

            # 1.8 seconds
            out = sp.permute(A, dim=dim, perms=perms, broadcast_perms_dim=broadcast_perms_dim, backward_mode=False)            

            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + A.dim() )
                ctx.broadcast_perms_dim = broadcast_perms_dim

                # Try to save space by converting perms to the smallest integer type that can represent it
                perms_max = A.shape[dim]-1

                if fsw_embedding_debug_mode:
                    assert perms_max == perms.max() # Sanity check

                if perms_max <= torch.iinfo(torch.int16).max:
                    perms = perms.to(dtype=torch.int16)
                elif perms_max <= torch.iinfo(torch.int32).max:
                    perms = perms.to(dtype=torch.int32)

                ctx.perms = perms
                ctx.nvals_A = A.values().numel()

            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            if ctx.needs_input_grad[2]:
                assert False, 'This is strange. Autograd requires gradient of permute_sparse() wrt the permutations'

            if not ctx.needs_input_grad[0]:
                return None, None, None, None
            
            dim = ctx.dim
            broadcast_perms_dim = ctx.broadcast_perms_dim
            perms = ctx.perms
            nvals_grad_output = grad_output._values().numel()
            nvals_A = ctx.nvals_A

            # Note: The output of the forward() of this function is used in four other functions:
            #       add_same_pattern, mul_same_pattern, mul_sparse_dense, cumsum_sparse
            #       and thus grad_output here is the sum of the four different grad_inputs returned by the backward() of these functions.
            #       It is likely that torch's autograd does not take into account that these four gradients are coalesced and have the same nonzero pattern,
            #       and thus we might end up here with an uncoalesced gradient, possibly with repeated entries.
            #       This happens in my tests, but two-fold rather than four-fold. When this is indeed the case, the two parts are united manually in the code below.
            #       Until there will be some way to explicitly implement that part of autograd's summation and exploit the fact that the different gradients
            #       are coalesced and have the same nonzero pattern. Or implement a class that inherits from torch's sparse tensors and implements custom
            #       A + B operator that assumes that A,B are coalesced and have the same nonzero pattern (and verifies it in debug mode).

            # TODO: Potential speedup and memory efficiency improvement: 
            #       Consider moving from torch's sparse tensors to manual (indices,values) representation, which would make autograd produce the correct result
            #       automatically. This will obviate the need to save two to four different copies of the indices of this huge tensor.

            if grad_output.is_coalesced():
                pass
            elif nvals_grad_output % nvals_A == 0:
                grad_output = sp.coalesce_repeated(grad_output, nvals_A)
            else:
                if fsw_embedding_debug_mode:
                    raise RuntimeError('Strange. Check why this happens')
                grad_output = grad_output.coalesce()
                
            out = sp.permute(grad_output, dim=dim, perms=perms, broadcast_perms_dim=broadcast_perms_dim, backward_mode=True)
            sp.verify_coalescence(out)

            return out, None, None, None



    # Permutes the values of a sparse tensor according to the input permutation
    class permute_sparse_vals(torch.autograd.Function):
        @staticmethod
        def forward(ctx, sparse_tensor, perm):
            sp.verify_coalescence(sparse_tensor, dense_dim=None)

            # Extract the indices and values
            indices = sparse_tensor.indices()
            values = sparse_tensor.values()
            
            # Permute the values according to the input permutation
            permuted_values = values[perm]
            del values
            
            # Create a new sparse tensor with permuted values
            output_sparse_tensor = sp.sparse_coo_tensor_coalesced(indices, permuted_values, sparse_tensor.size())
            del indices, permuted_values

            # Save the permutation for backward
            ctx.nvals = sparse_tensor.indices().shape[1]
            ctx.perm = perm if ctx.needs_input_grad[0] else None

            return output_sparse_tensor

        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve the permutation
            perm = ctx.perm            

            grad_output = sp.coalesce_repeated(grad_output, window_size=ctx.nvals)

            # The gradient w.r.t. the input values is the permuted gradient output
            grad_input_values = torch.empty_like(grad_output.values())
            grad_input_values[perm] = grad_output.values()
            
            # Create a gradient sparse tensor
            grad_input = sp.sparse_coo_tensor_coalesced(grad_output.indices(), grad_input_values, grad_output.shape)
            
            # No gradient for the permutation itself (None)
            return grad_input, None



    # Equivalent to: torch.unsqueeze(dim)
    class unsqueeze_sparse(torch.autograd.Function):
        @staticmethod        
        def forward(ctx, A, dim):
            assert_coalesced(A)

            # Note that here it is dim+A.dim()+1 rather than the usual dim + A.dim(), as defined in the unsqueeze() command
            ctx.dim = dim if dim >= 0 else ( dim + A.dim() + 1 )
            ctx.input_shape = A.shape

            vals = A.values()            
            inds = A.indices()
            inds = torch.cat( (inds[0:ctx.dim,:], torch.zeros((1,inds.shape[1]), device=inds.device, dtype=inds.dtype), inds[ctx.dim:,:]), dim=0 )
            
            output_shape = ctx.input_shape[0:ctx.dim] + (1,) + ctx.input_shape[ctx.dim:]

            out = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=output_shape)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            dim = ctx.dim          
            input_shape = ctx.input_shape

            input_dims_in_output = torch.arange(len(input_shape)+1, device=grad_output.device)
            input_dims_in_output = input_dims_in_output[input_dims_in_output != dim]

            inds = torch.index_select(grad_output.indices(), 0, input_dims_in_output)
            vals = grad_output.values()

            grad_input = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=input_shape)

            return grad_input, None



    # Equivalent to: torch.sum(A, dim=dim, keepdim=True)
    class sum_sparseToDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim, slice_info):
            assert A.is_sparse, 'A must be sparse'
            assert A.is_coalesced(), 'A must be coalesced'

            out = sp.sum_sparse(A, dim=dim, slice_info=slice_info).to_dense()

            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + A.dim() )
                ctx.shape = A.shape
                ctx.A_inds = A.indices()

            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            assert not grad_output.is_sparse, "This shouldn't happen"

            dim = ctx.dim
            shape = ctx.shape
            A_inds = ctx.A_inds
           
            vals = grad_output.expand(shape)[tuple(A_inds)]
            grad_input = sp.sparse_coo_tensor_coalesced(indices=A_inds, values=vals, size=shape)

            sp.verify_coalescence(grad_input)
            return grad_input, None, None



    # Given two sparse tensors A,B and two scalars a,b, compute a*A + b*B
    class add_same_pattern(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, a, b):
            assert_coalesced(A)
            assert_coalesced(B)
            sp.verify_coalescence(A)
            sp.verify_coalescence(B)

            if a == 1.0:
                out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=torch.add(A.values(), B.values(), alpha=b), size=A.shape)
            elif a == -1.0:
                out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=torch.add(A.values().neg(), B.values(), alpha=b), size=A.shape)
            else:
                out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=torch.add(a*A.values(), B.values(), alpha=b), size=A.shape)

            assert not (ctx.needs_input_grad[2] or ctx.needs_input_grad[3]), 'gradients of a,b are currently not implemented'

            ctx.a = a
            ctx.b = b

            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            grad_A = None
            grad_B = None

            a = ctx.a
            b = ctx.b

            if True in ctx.needs_input_grad:
                assert_coalesced(grad_output)

            if ctx.needs_input_grad[0]:
                if a == 1.0:
                    grad_A = grad_output.clone()
                else:
                    grad_A = sp.sparse_coo_tensor_coalesced(indices=grad_output.indices(), values=a*grad_output.values(), size=grad_output.shape)
            
            if ctx.needs_input_grad[1]:
                if b == 1.0:
                    grad_B = grad_output.clone()
                else:
                    grad_B = sp.sparse_coo_tensor_coalesced(indices=grad_output.indices(), values=b*grad_output.values(), size=grad_output.shape)

            sp.verify_coalescence(grad_A)
            sp.verify_coalescence(grad_B)
            return grad_A, grad_B, None, None



    # Multiply two sparse tensors that have the same nonzero pattern
    class mul_same_pattern(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, fac):
            assert_coalesced(A)
            assert_coalesced(B)
            
            sp.verify_coalescence(A)
            sp.verify_coalescence(B)

            if fac == 1.0:
                out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=A.values()*B.values(), size=A.shape)
            else:
                out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=fac*A.values()*B.values(), size=A.shape)

            grad_mult_A = None
            grad_mult_B = None

            if ctx.needs_input_grad[0]:
                grad_mult_A = B.values() if fac == 1.0 else fac * B.values()
            
            if ctx.needs_input_grad[1]:
                grad_mult_B = A.values() if fac == 1.0 else fac * A.values()

            assert not ctx.needs_input_grad[2], 'gradient of fac is currently not implemented'

            if True in ctx.needs_input_grad:
                ctx.save_for_backward(grad_mult_A, grad_mult_B)

            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            grad_mult_A, grad_mult_B = ctx.saved_tensors

            grad_A = None
            grad_B = None

            if True in ctx.needs_input_grad:
                assert_coalesced(grad_output)

            if ctx.needs_input_grad[0]:
                grad_A = sp.sparse_coo_tensor_coalesced(indices=grad_output.indices(), values=grad_mult_A*grad_output.values(), size=grad_output.shape)
            
            if ctx.needs_input_grad[1]:
                grad_B = sp.sparse_coo_tensor_coalesced(indices=grad_output.indices(), values=grad_mult_B*grad_output.values(), size=grad_output.shape)

            sp.verify_coalescence(grad_A)
            sp.verify_coalescence(grad_B)
            return grad_A, grad_B, None



    # Equivalent to: A*B, where A is sparse, B is dense, and the shape of B is broadcastable to A
    class mul_sparse_dense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, slice_info):    

            assert A.is_sparse, 'A must be sparse'
            assert not B.is_sparse, 'B cannot be sparse'
            assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
            assert A.is_coalesced(), 'A must be coalesced'

            sp.verify_coalescence(A)

            # Dimensions along B needs to be broadcast to A
            broadcast_dims = sp.get_broadcast_dims_B_to_A(A,B)
            vals = A.values() * B.expand_as(A)[tuple(A.indices())]

            out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            A_save = A if ctx.needs_input_grad[1] else None
            B_save = B if ctx.needs_input_grad[0] else None

            if True in ctx.needs_input_grad:
                ctx.save_for_backward(A_save, B_save)
                ctx.broadcast_dims = broadcast_dims
                ctx.slice_info = slice_info if ctx.needs_input_grad[1] else None 

            sp.verify_coalescence(out)
            return out

        # 1.65 seconds
        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            A, B = ctx.saved_tensors
            broadcast_dims = ctx.broadcast_dims
            slice_info = ctx.slice_info

            out_A = out_B = None

            if ctx.needs_input_grad[0]:
                # The to_sparse() below is just to make sure that B, which is dense, doesn't get broadcast
                # to the size of grad_output; it is the same size of A, which can be huge.

                assert grad_output.is_sparse
                assert_coalesced(grad_output)

                inds = grad_output.indices()            
                vals = grad_output.values() * B.expand_as(grad_output)[tuple(inds)]
                out_A = sp.sparse_coo_tensor_coalesced(indices=grad_output.indices(), values=vals, size=grad_output.shape)                

            if ctx.needs_input_grad[1]:
                # B is dense, so the gradient with respect to B should also be dense.
                if len(broadcast_dims) > 0:
                    # 0.04 seconds
                    out_B = sp.same_shape_prod(A, grad_output)
                        
                    out_B = sp.sum_sparse(out_B, dim=broadcast_dims, slice_info=slice_info).to_dense()
                else:
                    # 0 seconds
                    # In this case, B didn't need to be broadcast to A, so all of A,B,grad_output have the same shape and are not huge
                    out_B = sp.same_shape_prod(A, grad_output).to_dense()

            sp.verify_coalescence(out_A)
            return out_A, out_B, None



    # Equivalent to: A/B, where A is sparse, B is dense, and the shape of B is broadcastable to A
    class div_sparse_dense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, slice_info): # 0.06 seconds
            assert A.is_sparse, 'A must be sparse'
            assert not B.is_sparse, 'B cannot be sparse'
            assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
            assert A.is_coalesced(), 'A must be coalesced'

            sp.verify_coalescence(A)

            if fsw_embedding_debug_mode:
                assert (B > 0).all(), 'B cannot contain zeros'

            broadcast_dims = sp.get_broadcast_dims_B_to_A(A,B) # tuple(torch.nonzero(torch.tensor(B.shape) == 1))

            vals = A.values() / B.expand_as(A)[tuple(A.indices())]

            out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            A_save = A if ctx.needs_input_grad[1] else None
            B_save = B if True in ctx.needs_input_grad else None

            if True in ctx.needs_input_grad:
                ctx.save_for_backward(A_save, B_save)
                ctx.broadcast_dims = broadcast_dims
                ctx.slice_info = slice_info

            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output): # 0 seconds
            A, B = ctx.saved_tensors
            broadcast_dims = ctx.broadcast_dims
            slice_info = ctx.slice_info

            assert_coalesced(grad_output)

            out_A = out_B = None
        
            if ctx.needs_input_grad[0]:                
                out_A = sp.div_sparse_dense(grad_output, B)
            
            if ctx.needs_input_grad[1]:
                # B is dense, so the gradient with respect to B should also be dense.
                # both cases below take 0 seconds
                if len(broadcast_dims) > 0:
                    # 0 seconds
                    out_B = sp.same_shape_prod(A, grad_output) # This is still sparse and can be huge
                    out_B = sp.sum_sparse(out_B, dim=broadcast_dims, slice_info=slice_info).to_dense()
                    out_B = -out_B / torch.square(B)
                else:
                    # In this case, B didn't need to be broadcast to A, so all of A,B,grad_output have the same shape and are not huge
                    out_B = sp.same_shape_prod(A, grad_output).to_dense() / (-torch.square(B))
                    assert False, "Did not test this case. Make sure this works correctly"

            sp.verify_coalescence(out_A)
            return out_A, out_B, None



    class add_sparse_dense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B, slice_info):
            assert A.is_sparse, 'A must be sparse'
            assert not B.is_sparse, 'B cannot be sparse'
            assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
            assert A.is_coalesced(), 'A must be coalesced'

            # Calculate broadcast dimensions
            broadcast_dims = sp.get_broadcast_dims_B_to_A(A, B)

            # Perform entrywise addition
            vals = A.values() + B.expand_as(A)[tuple(A.indices())]

            # Create the output sparse tensor with the same indices as A
            out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            if True in ctx.needs_input_grad:
                ctx.broadcast_dims = broadcast_dims
                ctx.slice_info = slice_info if ctx.needs_input_grad[1] else None 

            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            broadcast_dims = ctx.broadcast_dims
            slice_info = ctx.slice_info

            out_A = out_B = None

            if ctx.needs_input_grad[0]:
                # Gradient with respect to the sparse tensor A is simply the grad_output
                out_A = grad_output
            
            if ctx.needs_input_grad[1]:
                # Gradient with respect to the dense tensor B
                if len(broadcast_dims) > 0:
                    # If broadcasting happened, we need to sum over the broadcast dimensions
                    out_B = sp.sum_sparse(grad_output, dim=broadcast_dims, slice_info=slice_info).to_dense()
                else:
                    # No broadcasting needed, so just convert to dense
                    out_B = grad_output.to_dense()

            sp.verify_coalescence(out_A)
            return out_A, out_B, None


    # Calculates the function f(x) = max(x, thresh), but with grad f(thresh) = 1
    class custom_lowclamp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, thresh):
            ctx.active = (input >= thresh)
            return torch.clamp(input, min=thresh)
        
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.where(ctx.active, grad_output, torch.zeros_like(grad_output))
            return grad_input, None


    # Equivalent to torch.sinc(A) * torch.cos(B), where A and B are spase tensors of the same shape and nonzero pattern.
    class sinc_cos_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            assert A.is_sparse and B.is_sparse
            assert (A.shape == B.shape), 'A and B must have the same shape'
            assert A.is_coalesced(), 'A must be coalesced'
            assert B.is_coalesced(), 'B must be coalesced'

            sp.verify_coalescence(A)
            sp.verify_coalescence(B)

            if fsw_embedding_debug_mode:
                assert (A.indices() == B.indices()).all(), 'A and B nonzero indices do not match'

            inds = A.indices()

            A_vals = A.values()
            B_vals = B.values()

            B_cos = torch.cos(B_vals)

            if ctx.needs_input_grad[0]:
                A_dsinc, A_sinc = sp.dsinc(A_vals, return_sinc=True)
                grad_mult_A = A_dsinc*B_cos
            else:
                A_sinc = torch.sinc(A_vals)
                A_dsinc = None
                grad_mult_A = None

            if ctx.needs_input_grad[1]:
                B_sin = torch.sin(B_vals)
                grad_mult_B = A_sinc * (-B_sin)
                del B_sin
            else:
                grad_mult_B = None

            out_vals = A_sinc*B_cos

            if True in ctx.needs_input_grad:
                ctx.save_for_backward(grad_mult_A, grad_mult_B)
                ctx.shape = A.shape
                ctx.inds = inds

            out = sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals, size=A.shape)
            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            grad_mult_A, grad_mult_B = ctx.saved_tensors
            shape = ctx.shape
            inds = ctx.inds

            grad_A = grad_B = None

            if ctx.needs_input_grad[0]:
                grad_vals_A = grad_mult_A * grad_output.values()
                grad_A = sp.sparse_coo_tensor_coalesced(indices=inds, values=grad_vals_A, size=shape)
                del grad_vals_A

            if ctx.needs_input_grad[1]:
                grad_vals_B = grad_mult_B * grad_output.values()
                grad_B = sp.sparse_coo_tensor_coalesced(indices=inds, values=grad_vals_B, size=shape)
                del grad_vals_B

            sp.verify_coalescence(grad_A)
            sp.verify_coalescence(grad_B)
            return grad_A, grad_B


    # Concatenates two sparse tensors along their last sparse dimension, similarly to torch.cat((A,B), dim=A.sparse_dim()-1)
    # Supports tensors with dense_dim > 0. The two tensors should have the same dense_dim() and have the same shape along all dimensions
    # except for sparse_dim()-1
    class concat_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            # Ensure both A and B are sparse and coalesced
            sp.verify_coalescence(A, dense_dim=None)
            sp.verify_coalescence(B, dense_dim=None)
            assert (A.dim() == B.dim()) and (A.sparse_dim() == B.sparse_dim()), "A and B must have the same shape, except for their last sparse dimension, which should be identical"
            sA = replace_in_tuple(A.shape, A.sparse_dim()-1, 1)
            sB = replace_in_tuple(B.shape, B.sparse_dim()-1, 1)
            assert (sA == sB), "A and B must have the same shape, except for their last sparse dimension, which should be identical"

            sparse_dim = A.sparse_dim()
            dense_dim = A.dense_dim()
            A_nvals = A.indices().shape[1]
            B_nvals = B.indices().shape[1]

            # This is identical for A and B
            dense_shape = A.shape[sparse_dim:]

            # Calculate the offset for indices of B
            offset = A.shape[sparse_dim-1]

            # Concatenate indices and values
            if True:
                # More efficient variant
                combined_indices = torch.empty((sparse_dim, A_nvals+B_nvals), device=A.device, dtype=A.indices().dtype)
                combined_indices[:, 0:A_nvals] = A.indices()
                combined_indices[:, A_nvals:] = B.indices()
                combined_indices[-1, A_nvals:] += offset

                combined_values = torch.empty( (A_nvals+B_nvals,) + dense_shape, device=A.device, dtype=A.dtype)
                combined_values[0:A_nvals, ...] = A.values()
                combined_values[A_nvals:, ...] = B.values()
            else:
                # Clearer variant
                new_B_indices = B_indices.clone()
                new_B_indices[-1, :] += offset
                combined_indices = torch.cat([A.indices(), new_B_indices], dim=1)
                combined_values = torch.cat([A.values(), B.values()], dim=0)

            # Get the size of the resulting sparse tensor
            combined_size = list(A.shape)
            combined_size[sparse_dim-1] += B.shape[sparse_dim-1]

            # Create the output sparse tensor and coalesce it
            combined_indices, combined_values = sp.sort_inds_vals(combined_indices, combined_values, combined_size) 
            C = sp.sparse_coo_tensor_coalesced(combined_indices, combined_values, combined_size)

            # Save information for backward pass
            ctx.sparse_dim = sparse_dim
            ctx.dense_dim = dense_dim
            ctx.A_shape = A.shape
            ctx.B_shape = B.shape

            sp.verify_coalescence(C, dense_dim=dense_dim)
            return C

        @staticmethod
        def backward(ctx, grad_output):
            sparse_dim = ctx.sparse_dim
            dense_dim = ctx.dense_dim
            A_shape = ctx.A_shape
            B_shape = ctx.B_shape

            if not (True in ctx.needs_input_grad):
                return None, None

            sp.verify_coalescence(grad_output, dense_dim=dense_dim), 'If for some reason we get an uncoalesced grad_output here, we should check why and if justified, coalesce it.'

            # Extracting the indices and values from grad_output
            grad_indices = grad_output.indices()
            grad_values = grad_output.values()

            if ctx.needs_input_grad[0]:
                A_mask = grad_indices[-1, :] < A_shape[sparse_dim-1]
                grad_A_indices = grad_indices[:, A_mask]
                grad_A_values = grad_values[A_mask, ...]
                grad_A = sp.sparse_coo_tensor_coalesced(grad_A_indices, grad_A_values, size=A_shape)
                del A_mask, grad_A_indices, grad_A_values
            else:
                grad_A = None

            if ctx.needs_input_grad[1]:
                B_mask = grad_indices[-1, :] >= A_shape[sparse_dim-1]
                grad_B_indices = grad_indices[:, B_mask]
                grad_B_indices[-1, :] -= A_shape[-1]
                grad_B_values = grad_values[B_mask, ...]
                grad_B = sp.sparse_coo_tensor_coalesced(grad_B_indices, grad_B_values, size=B_shape)
                del B_mask, grad_B_indices, grad_B_values
            else:
                grad_B = None

            sp.verify_coalescence(grad_A, dense_dim=dense_dim)
            sp.verify_coalescence(grad_B, dense_dim=dense_dim)
            return grad_A, grad_B



    # Takes a sparse tensor whose dense dimension is 0 and returns a sparse tensor with the same shape, whose dense dimension is 1.
    class unsqueeze_dense_dim(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            sp.verify_coalescence(A, dense_dim=0)
            
            # Save context for backward pass
            ctx.A_shape = tuple(A.shape)
            
            # Extract indices and values
            indices = A.indices()  # Shape: (sparse_dim, nvals)
            values = A.values()    # Shape: (nvals,)                
            
            # Reshape values to have a dense dimension of 1
            values_out = values.view(-1, 1)
            
            # Create the output sparse tensor            
            output_sparse_tensor = sp.sparse_coo_tensor_coalesced(indices, values_out, ctx.A_shape + (1,))
            
            return output_sparse_tensor

        @staticmethod
        def backward(ctx, grad_output):
            if not ctx.needs_input_grad[0]:
                return None
            
            sp.verify_coalescence(grad_output, dense_dim=1)

            indices = grad_output.indices()
            values = grad_output.values()

            # Reverse the transformation: remove the added dense dimension
            values_out = values.view(-1)  # Flatten the values
            
            grad_input = sp.sparse_coo_tensor_coalesced(indices, values_out, ctx.A_shape)
            
            return grad_input
        


    # Takes a sparse tensor whose dense dimension is 1 and returns a sparse tensor with the same shape, whose dense dimension is zero.
    class flatten_dense_dim(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            sp.verify_coalescence(A, dense_dim=1)
            
            # Save context for backward pass
            ctx.shape = A.shape
            
            # Extract indices and values
            indices = A.indices()  # Shape: (sparse_dim, nvals)
            values = A.values()    # Shape: (nvals, dense_dim)
            
            # Flatten the dense dimension into the sparse indices
            nvals_in = values.size(0)
            nvals_out = values.numel()
            dense_dim = values.size(1)            
            
            # Repeat indices for each dense element
            if True:
                # More efficient variant
                indices_out = torch.empty((A.dim(), nvals_out), device=A.device, dtype=A.indices().dtype)
                indices_out[0:-1, :] = indices.repeat_interleave(dense_dim, dim=1)
                indices_out[-1, :] = torch.arange(dense_dim, device=indices.device).repeat(nvals_in).view(1, -1)
            else:
                # Clearer variant
                expanded_indices = indices.repeat_interleave(dense_dim, dim=1)
                
                # Create new indices for the dense dimension
                additional_indices = torch.arange(dense_dim, device=indices.device).repeat(nvals_in)
                additional_indices = additional_indices.view(1, -1)
                
                # Combine sparse and new dense indices
                indices_out = torch.cat([expanded_indices, additional_indices], dim=0)
         
            # Flatten values
            values_out = values.view(-1)
            
            # Create the output sparse tensor            
            output_sparse_tensor = sp.sparse_coo_tensor_coalesced(indices_out, values_out, A.shape)
            
            return output_sparse_tensor

        @staticmethod
        def backward(ctx, grad_output):
            if not ctx.needs_input_grad[0]:
                return None
            
            sp.verify_coalescence(grad_output)

            indices = grad_output.indices()
            values = grad_output.values()

            # Reverse the transformation: reconstruct the gradient for the original dense dimension
            dense_dim = ctx.shape[-1]

            arange_indices = torch.arange(0, indices.shape[1], dense_dim, device=indices.device)
            reduced_indices = indices[:-1, arange_indices]
            del indices, arange_indices

            grad_values = values.view(-1, dense_dim)
            del values

            grad_input = sp.sparse_coo_tensor_coalesced(reduced_indices, grad_values, ctx.shape)
            
            return grad_input


    # Equivalent to torch.sort(X, dim=dim, descending=descending)
    # Works only on dense tensors
    class sort(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, dim, descending):
            assert not X.is_sparse
            Xs, Xi = torch.sort(X, dim=dim, descending=descending)
            
            # Store the dimension and sorting permutation for back propagation
            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + X.dim() )

                # Try to save space
                Xi_max = X.shape[dim]-1
                #assert Xi_max == Xi.max() # Sanity check

                if Xi_max <= torch.iinfo(torch.int16).max:
                    Xi = Xi.to(dtype=torch.int16)
                elif Xi_max <= torch.iinfo(torch.int32).max:
                    Xi = Xi.to(dtype=torch.int32)

                ctx.Xi = Xi

            return Xs, Xi


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output, aaa):
            dim = ctx.dim
            Xi = ctx.Xi

            if not ctx.needs_input_grad[0]:
                grad_input = None
            elif grad_output.is_sparse:
                assert False, 'This should not happen'
            else:
                # 0.089 seconds
                grad_input = torch.empty_like(grad_output)
                grad_input.scatter_(dim=dim, index=Xi.to(torch.int64), src=grad_output)

            return grad_input, None, None



    # Sorts a sparse tensor along a given dimension
    class sort_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim, descending):
            dim = dim if dim >= 0 else ( dim + A.dim() )

            sp.verify_coalescence(A, dense_dim=0)

            # Extract indices and values
            inds = A.indices()
            vals = A.values()
            
            sortdims = [d for d in range(A.dim()) if d != dim]

            inds2 = inds[sortdims,:]
            shape2 = [A.shape[d] for d in sortdims]

            inds1d = sp.ravel_index(inds2, shape2)
            del inds2, shape2

            perm_vals = torch.argsort(vals, descending=descending, stable=True)
            perm_inds = torch.argsort(inds1d[perm_vals], descending=False, stable=True)            

            if dim==A.dim()-1:
                del inds1d
                perm = perm_vals[perm_inds]
                del perm_vals, perm_inds
            else:                
                perm_inds2 = torch.argsort(inds1d, descending=False, stable=True)
                del inds1d
                perm = torch.empty_like(perm_vals)
                perm_dest = perm_vals[perm_inds]
                del perm_vals, perm_inds                
                perm[perm_inds2] = perm_dest
                del perm_inds2
                      
            # Reconstruct sorted sparse tensor
            out = sp.sparse_coo_tensor_coalesced(inds, vals[perm], A.shape)
            
            # Save the original indices and sorted indices for backward
            if ctx.needs_input_grad[0]:
                ctx.perm = perm
            
            return out, perm

        @staticmethod
        def backward(ctx, grad_output, aaa):

            if ctx.needs_input_grad[0]:
                sp.verify_coalescence(grad_output)

                inds = grad_output.indices()
                vals = grad_output.values()

                vals_out = torch.empty(vals.shape[0], device=grad_output.device, dtype=grad_output.dtype)
                vals_out[ctx.perm] = vals
                del vals
                grad_input = sp.sparse_coo_tensor_coalesced(inds, vals_out, grad_output.shape)
                del inds
            else:
                grad_input = None
                      
            return grad_input, None, None
    


    # Equivalent to torch.cumsum(X, dim=dim)
    class cumsum_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, dim, slice_info):
            assert X.is_coalesced(), 'X must be coalesced'

            sp.verify_coalescence(X)

            out = sp.cumsum_sparse(X, dim=dim, slice_info=slice_info)            

            if ctx.needs_input_grad[0]:
                ctx.dim = dim
                ctx.slice_info = slice_info

            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):            
            if not ctx.needs_input_grad[0]:
                return None, None, None

            sp.verify_coalescence(grad_output)

            dim = ctx.dim
            slice_info = ctx.slice_info
            grad_input = sp.cumsum_sparse(grad_output, dim=dim, slice_info=slice_info, reverse=True) 

            sp.verify_coalescence(grad_input)

            return grad_input, None, None



    # Replicates the sparse tensor A n times along the dimension dim
    # Worsk fastest if dim=0 or dim=-1; see comments inside.
    class repmat_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, n, dim):
            dim = dim if dim >= 0 else ( dim + A.dim() )
            assert (dim >= 0) and (dim < A.dim())
            
            ctx.dim = dim
            ctx.shape = A.shape

            dim = ctx.dim

            assert_coalesced(A)
            sp.verify_coalescence(A)

            # If dim=0, then variant 1 keeps the output tensor sorted.
            # Similarly, if dim=-1, then variant 2 keeps the output tensor sorted.
            # So in these cases, there is no need to call coalesce().
            # Variant 2 is negligibly faster than 1.
            
            # Changing this choice might break the correctness of the code.
            variant = 2 if dim == len(A.shape)-1 else 1

            if variant == 1:
                v = A.shape[dim] * torch.arange(n, device=A.device)
                v = torch.repeat_interleave(v, A.values().numel(), dim=0)                

                inds = A.indices().repeat([1,n])                
                inds[dim,:] += v
                del v

                vals = A.values().repeat([n,])

                out_shape = list(A.shape)
                out_shape[dim] = n*out_shape[dim]

            elif variant == 2:
                v = A.shape[dim] * torch.arange(n, device=A.device)
                v = v.repeat(A.values().numel())

                inds = A.indices().repeat_interleave(repeats=n, dim=1)

                inds[dim,:] += v
                del v

                vals = A.values().repeat_interleave(repeats=n, dim=0)

                out_shape = list(A.shape)
                out_shape[dim] = n*out_shape[dim]

            del A

            # If dim=-1 or 0, and the correct variant was used, then inds are already sorted.
            if dim not in (0, len(ctx.shape)-1):
                inds, vals = sp.sort_inds_vals(inds, vals, out_shape)

            out = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=out_shape)
            sp.verify_coalescence(out)
            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            if ctx.needs_input_grad[0]:
                assert_coalesced(grad_output)
                grad_output = grad_output
                inds = grad_output.indices().clone()
                inds[ctx.dim] = torch.remainder(inds[ctx.dim], ctx.shape[ctx.dim])

                # TODO: Maybe this can be sped up when dim=0 or dim=-1 because we can calculate the sum explicitly instead of calling coalesce()
                # 0 seconds
                grad_input = torch.sparse_coo_tensor(indices=inds, values=grad_output.values(), size=ctx.shape)
                # Here we must coalesce
                grad_input = grad_input.coalesce()

            else:
                grad_input = None

            sp.verify_coalescence(grad_input)
            return grad_input, None, None




#############################################################################################################
##                                        Sparse tensor operations                                         ##
#############################################################################################################

class sp:
    # Create a COO sparse tensor in a coalesced state, assuming that the input indices are already coalesced.
    # If the command sp.verify_coalescence(out) is not commented, the tensor is verified for being correctly coalesced.
    def sparse_coo_tensor_coalesced(indices, values, size:list[int]):
        sparse_dims = indices.shape[0]
        if fsw_embedding_debug_mode:
            inds1d = sp.ravel_index(indices, shape=size[0:sparse_dims])
            assert torch.unique(inds1d).numel() == inds1d.numel(), 'indices are not unique'
            assert (torch.sort(inds1d)[0] == inds1d).all(), 'indices are unique but not sorted'

        out = torch.sparse_coo_tensor(indices=indices, values=values, size=size, is_coalesced=True)
        sp.verify_coalescence(out, dense_dim=out.dense_dim())
        return out   



    # returns a coalesced copy of A assuming that the indices of A are unique but just possibly unsorted
    def coalesce_unique(A):
        A_shape = A.shape
        inds2, vals2 = sp.sort_inds_vals(indices=A._indices(), values=A._values(), shape=A.shape)
        del A
        return sp.sparse_coo_tensor_coalesced(inds2, vals2, size=A_shape)



    # returns a coalesced copy of A assuming that A contains several repetitions of the same set of indices, with each set being already coalesced
    def coalesce_repeated(A, window_size):
        assert A._indices().shape[1] % window_size == 0, 'coalesce_repeated: number of indices is not an integer multiple of copy_size'
        num_repeats = A._indices().shape[1] // window_size

        inds_out = A._indices()[:, 0:window_size]
        vals_out = A._values()[0:window_size, ...]

        for i in range(1, num_repeats):
            if fsw_embedding_debug_mode:
                assert (inds_out == A._indices()[:, (i*window_size):((i+1)*window_size)]).all(), 'coalesce_repeated: index mismatch between differnt windows'
            vals_out += A._values()[(i*window_size):((i+1)*window_size), ...]
        
        #inds2, vals2 = sp.sort_inds_vals(indices=A._indices(), values=A._values(), shape=A.shape)
        #del A
        return sp.sparse_coo_tensor_coalesced(inds_out, vals_out, size=A.shape)



    # Verify that a sparse input tensor A is correctly coalesced.
    def verify_coalescence(A, dense_dim=0):
        if A is None:
            return
        
        assert A.is_sparse, 'verify_coalescence: input tensor is not sparse'
        assert (dense_dim is None) or (A.dense_dim() == dense_dim), 'verify_coalescence: incorrect dense_dim(); expected %d, got %d' % (dense_dim, A.dense_dim())
        assert A.is_coalesced(), 'verify_coalescence: input tensor is not coalesced'

        if fsw_embedding_debug_mode:            
            B = torch.sparse_coo_tensor(indices=A.indices(), values=A.values(), size=A.shape).coalesce()
            assert (B.indices() == A.indices()).all(), 'verify_coalescence: index mismatch in input'
            assert (B.values() == A.values()).all(), 'verify_coalescence: value mismatch in input'



    # Takes an input dense tensor A and returns a sparse tensor that contains *all* entries in A, including zeros
    def to_sparse_full(A):
        assert not A.is_sparse, 'A should be dense'

        # Get the indices of all elements
        indices = torch.nonzero(torch.ones_like(A), as_tuple=False).t()

        # Flatten the dense tensor to create values for all elements, including zeros
        values = A.flatten()

        # Create the sparse tensor, including all values
        out = sp.sparse_coo_tensor_coalesced(indices, values, size=A.shape)

        return out



    def same_shape_prod(A,B):
        assert A.shape == B.shape
        assert A.values().numel() == B.values().numel(), 'A, B have different numbers of nonzero values'
        if fsw_embedding_debug_mode:
            assert (A.indices() == B.indices()).all(), 'A, B nonzero pattern mismatch'
        out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=A.values()*B.values(), size=A.shape)
        return out



    def same_shape_prod_(A,B):
        assert A.shape == B.shape
        assert A.values().numel() == B.values().numel(), 'A, B have different numbers of nonzero values'
        if fsw_embedding_debug_mode:
            assert (A.indices() == B.indices()).all(), 'A, B nonzero pattern mismatch'
        A_vals = A.values()
        A_vals *= B.values()
        return A



    # The inverse of torch.unravel_index()
    # Torch JIT does not speed up this function
    # 1.08 seconds
    def ravel_index(indices, shape):
        assert indices.dim() == 2, 'indices must be a 2-dimensional tensor'
        nd = indices.shape[0]

        assert nd > 0, 'input indices are of shape 0 x <something>'

        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, device=indices.device, dtype=indices.dtype)

        weights = shape.reshape([nd,1]).flip(dims=(0,))[0:-1].cumprod(dim=0).flip(dims=(0,))
        weights = torch.cat((weights, torch.ones(size=(1,1), device=weights.device, dtype=weights.dtype)), dim=0)

        # Variants 2 and 3 use torch matrix/tensor multiplication, but they return the error:
        # RuntimeError: "addmm_cuda" not implemented for 'Long'        
        variant = 1

        if variant == 1:
            # 0.7 seconds, but no better way to do it
            out = torch.sum(indices*weights, dim=0) 
        elif variant == 2:
            out = ( weights.transpose(0,1) @ indices ).view(-1)
        elif variant == 3:
            out = torch.tensordot(indices, weights, dims=((0,),(0,))).view(-1)

        return out



    # Sort indices and values. Similar to coalesce(), but does not assume nor impose uniqueness.
    # 2.93 seconds
    def sort_inds_vals(indices, values, shape=None):
        dense_dims = values.dim()-1
        sparse_dims = indices.shape[0]

        if shape is None:
            shape, _ = torch.max(indices, dim=1)
            shape += 1
            shape = tuple(shape)
        else:
            shape = tuple(shape)
            shape = shape[0:sparse_dims]        

        # 0.54 seconds
        inds1d = sp.ravel_index(indices, shape)

        debug = fsw_embedding_debug_mode
        if debug:
            assert ( len(torch.unique(inds1d)) == len(inds1d) ), 'indices are not unique'

        # 1.8 seconds
        sort_perm = torch.argsort(inds1d)

        del inds1d

        # 0.6 seconds
        if sort_perm.dim() == 1:
            out = (indices[:,sort_perm], values[sort_perm])
        else:
            out = (indices[:,sort_perm], values[sort_perm,:])

        if fsw_embedding_debug_mode:
            assert (out[0] != indices).any(), 'input was already sorted. the code can be sped up by avoiding sort in this particular case'

        return out



    # Entrywise division of sparse A by dense A. Supports broadcasting of B to A.
    def div_sparse_dense(A,B): 
        assert A.is_sparse, 'A must be sparse'
        assert not B.is_sparse, 'B must be dense'
        assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
        assert A.is_coalesced(), 'A must be coalesced'
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'

        inds = A.indices()
        A_vals = A.values()

        out_vals = A_vals / B.expand_as(A)[tuple(inds)]

        out = sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals, size=A.shape)
        return out



    def sum_sparse(A, dim, slice_info=None):

        assert A.is_sparse, "input tensor must be sparse"
        assert_coalesced(A)

        if fsw_embedding_debug_mode:
            assert slice_info is not None, 'if this happens, there might be an inefficiency in the code'

        # Shape of the sparse tensor
        shape = list(A.shape)
        dim = sp.dim_to_list(shape, dim)

        if slice_info is None:
            slice_info = sp.get_slice_info(A, dim)            

        # Verify slice info matches the input tensor
        sp.verify_slice_info(A, dim, slice_info)        

        si = slice_info

        inds = A.indices()
        vals = A.values()
        
        # Sort the values according to the keys, to form contiguous segments
        vals_sorted = vals[si['sort_inds']]

        # # Get the linear index in vals_sorted of the last index of each slice along dims
        # slice_ends = counts_consecutive.cumsum(dim=0)-1

        if si['num_slices'] > 1:
            vals_sorted_cumsum = segcumsum(vals_sorted, si['keys_sorted'], in_place=True, max_seg_size=si['max_slice_nonzeros'], thorough_verify_input=fsw_embedding_debug_mode)
        else:
            # If there is only one slice, we don't need to use segmented cumsum
            vals_sorted_cumsum = torch.cumsum(vals_sorted, dim=0, out=vals_sorted)

        del vals_sorted

        # Calculate output shape
        shape_out = list(A.shape)

        for d in dim:
            shape_out[d] = 1

        # Prepare output values and indices
        vals_out = vals_sorted_cumsum[si['slice_ends']]

        inds_out = inds[:, si['sort_inds'][si['slice_ends']]]
        inds_out[dim,:] = 0

        # Create a new sparse tensor with cumulative sum values
        out = sp.sparse_coo_tensor_coalesced(indices=inds_out, values=vals_out, size=shape_out)

        return out        



    # 1.2 seconds
    # Computes the cumsum on the nonzero entries of a sparse tensor A along dimension dim
    # max_slice_nonzers: An upper bound on the maximal number of nonzeros of A along dimension dim, taken over all slices of A along dim
    def cumsum_sparse(A, dim, slice_info=None, reverse=False):
        assert A.is_sparse, "input tensor must be sparse"        
        assert_coalesced(A)
        
        assert slice_info is not None, 'if this happens, there might be an inefficiency in the code'

        inds = A.indices()
        vals = A.values()

        # Shape of the sparse tensor
        shape = list(A.shape)

        dim = dim if dim >= 0 else ( dim + A.dim() )
        assert (dim >= 0) and (dim < inds.shape[0])

        # Get the other dimensions excluding the one we're summing over,
        # and get the shape along these dimensions.
        dims2 = [d for d in range(len(shape)) if d != dim]
        shape2 = [shape[d] for d in range(len(shape)) if d != dim]

        if slice_info is None:
            slice_info = sp.get_slice_info(A, dim)
        
        # Verify slice info matches the input tensor
        sp.verify_slice_info(A, dim, slice_info)        
          
        if reverse:
            keys_sorted = torch.flip(slice_info['keys_sorted'], [0,])
            sort_inds = torch.flip(slice_info['sort_inds'], [0,])
        else:
            keys_sorted = slice_info['keys_sorted']
            sort_inds = slice_info['sort_inds']

        max_slice_nonzeros = slice_info['max_slice_nonzeros']

        # 0.05 seconds
        # Sort the values according to the keys, to form contiguous segments
        vals_sorted = vals[sort_inds]

        if slice_info['num_slices'] > 1:
            vals_sorted_cumsum = segcumsum(vals_sorted, keys_sorted, in_place=True, max_seg_size=max_slice_nonzeros,
                                       thorough_verify_input=fsw_embedding_debug_mode)
        else:
            # If there is only one slice, we don't need to use segmented cumsum
            vals_sorted_cumsum = torch.cumsum(vals_sorted, dim=0, out=vals_sorted)

        del vals_sorted

        # 0.05 seconds
        vals_out = torch.empty_like(vals_sorted_cumsum)
        vals_out[sort_inds] = vals_sorted_cumsum

        # Create a new sparse tensor with cumulative sum values
        out = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals_out, size=shape)
        
        return out
        


    def sparse_flip(A, dim):
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'
        assert A.is_sparse
        assert A.is_coalesced()

        inds = A.indices().clone()
        vals = A.values()

        inds[dim, :] = A.shape[dim] - inds[dim, :] - 1

        inds2, vals2 = sp.sort_inds_vals(inds, vals, shape=A.shape)
        out = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=A.shape)
        return out



    def get_slice_info(A, dim, calc_nnz_per_slice=False):
        # We run with no gradients to ensure speed
        with torch.no_grad():
            assert A.is_sparse, "input tensor must be sparse"
            assert_coalesced(A)
            
            inds = A.indices()

            # Shape of the sparse tensor
            shape = list(A.shape)

            dim = sp.dim_to_list(shape, dim)
            
            # Get the other dimensions excluding the one we're summing over,
            # and get the shape along these dimensions.
            dims2 = [d for d in range(len(shape)) if not d in dim]
            shape2 = [shape[d] for d in range(len(shape)) if not d in dim]

            if len(dims2) == 0:
                if False:
                    keys = torch.zeros(A.values().numel(), device=inds.device, dtype=inds.dtype)            
                else:
                    # This variant is slightly faster than the avove, since it creates keys as an expanded scalar tensor.
                    # Note that consequently, keys will not be contiguous.
                    # To deal with it, sum_sparse should call cumsum() instead of segcumsum() in this case, since there is only one segment.
                    keys = torch.zeros(1, device=inds.device, dtype=inds.dtype).expand(A.values().numel())
            else:
                keys = sp.ravel_index(inds[dims2,:], tuple(shape2)).view(-1)


            variant = 1

            # No need to sort if len(dims2) == 0
            if len(dims2) == 0:
                keys_sorted = keys
                sort_inds = torch.arange(keys.numel(), device=keys.device, dtype=keys.dtype)
            elif variant == 1:
                # 2.74 seconds
                keys_sorted, sort_inds = torch.sort(keys, dim=0, stable=True)
            elif variant == 2:
                # Note: If there is a memory problem, try to use this variant
                sort_inds = torch.empty_like(keys)
                keys_sorted, sort_inds = torch.sort(keys, dim=0, stable=True, out=(keys, sort_inds))

            del keys

            # variant 2 is roughly twice faster
            variant = 2
            if len(dims2) == 0:
                one = torch.ones(1, device=keys_sorted.device, dtype=keys_sorted.dtype)
                slice_ends = (keys_sorted.numel()-1)*one
                slice_sizes = (keys_sorted.numel())*one
                max_slice_nonzeros = int(keys_sorted.numel())

            elif variant == 1:
                one = torch.ones(1, device=keys_sorted.device, dtype=keys_sorted.dtype)

                # Get the linear index in vals_sorted of the last index of each slice along dims
                # 0.52 seconds
                slice_ends = torch.nonzero(torch.diff(keys_sorted, n=1, dim=0, prepend=None, append=(keys_sorted.numel()-1)*one)).view(-1)
                # 0.1 seconds
                slice_sizes = torch.diff(slice_ends, n=1, dim=0, prepend=-one).view(-1)
                max_slice_nonzeros = int(torch.max(slice_sizes))

            elif variant == 2:
                _, counts_consecutive = torch.unique_consecutive(keys_sorted, return_counts=True)
                del _

                slice_ends = counts_consecutive.cumsum(dim=0)-1
                max_slice_nonzeros = int(torch.max(counts_consecutive))            

            slice_info = { 'shape': shape,
                        'dims': dim,
                        'keys_sorted': keys_sorted,
                        'sort_inds': sort_inds,
                        'slice_ends': slice_ends,
                        'num_slices': slice_ends.numel(),
                        'max_slice_nonzeros': max_slice_nonzeros}


            if calc_nnz_per_slice:
                # TODO: Make sure this works
                one = torch.ones(1, device=inds.device, dtype=torch.float64)
                vals = one.repeat(A.values().numel())
                A_mask = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=A.shape)
                nnz_per_slice = sp.sum_sparse(A_mask, dim, slice_info=slice_info).to_dense().to(torch.int64)
                del A_mask
                
                slice_info['nnz_per_slice'] = nnz_per_slice
            else: 
                slice_info['nnz_per_slice'] = None

        return slice_info



    def verify_slice_info(A, dim, slice_info):
        shape = list(A.shape)
        dim = dim if isinstance(dim, list) else sp.dim_to_list(shape, dim)

        assert shape == slice_info['shape'], 'slice_info is inconsistent with input tensor and dim'
        assert sorted(dim) == sorted(slice_info['dims']), 'slice_info is inconsistent with input tensor and dim'



    def dim_to_list(shape, dim):
        shape = list(shape)

        # Process dim
        if isinstance(dim, numbers.Number):
            dim = [dim,]
        else:            
            dim = list(dim)

        for i,d in enumerate(dim):
            dim[i] = d if d >= 0 else ( d + len(shape) )
            assert (dim[i] >= 0) and (dim[i] < len(shape))
        
        return dim



    # When B needs to be broadcast to A, returns the list of dimensions in B that need to be broadcast
    def get_broadcast_dims_B_to_A(A, B):
        assert A.dim() == B.dim(), 'A and B must have the same number of dimensions'
        ndims = A.dim()
        broadcast_dims = [d for d in range(ndims) if (B.shape[d] == 1) and (A.shape[d] > 1)]
        
        for d in range(ndims):
            assert (A.shape[d] == B.shape[d]) or (d in broadcast_dims), 'A must be of the same size as B, except for dimensions in B that equal 1'

        return broadcast_dims



    def permute(A, dim, perms, broadcast_perms_dim=None, backward_mode=False):
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'
        if broadcast_perms_dim is None:
            assert A.shape == perms.shape
        else:
            assert replace_in_tuple(tuple(A.shape), broadcast_perms_dim, 1) == perms.shape        

        sp.verify_coalescence(A)

        # Invert the given permutations.
        # Although the second option has linear time complexity, the first one is simpler and there is no difference in the actual running times, even on huge inputs.
        if backward_mode:
            # If we're in backward mode, do not invert
            perm_invs = perms
        else:
                # Variant 2 is faster
                variant = 2

                if variant == 1:
                    perm_invs = torch.argsort(perms, dim=dim)
                elif variant == 2: # 0.1 seconds
                    perm_invs = torch.empty_like(perms)
                    ar = torch.arange(perms.shape[dim], dtype=perms.dtype, device=perms.device)
                    ar = ar.reshape((1,)*dim + (len(ar),) + (1,)*(len(perms.shape)-(dim+1))).expand_as(perms)                    
                    perms = perms.to(torch.int64) # Note: This conversion is required for scatter_(), and it takes negligible time.
                    perm_invs.scatter_(dim=dim, index=perms, src=ar)
                    del ar

        inds = A.indices().clone()            
        inds[dim,:] = perm_invs.expand_as(A)[tuple(A.indices())]

        # 1.45 seconds
        inds2, vals2 = sp.sort_inds_vals(inds, A.values(), shape=A.shape)

        del inds, perms, perm_invs

        out = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=A.shape)
        return out



    # Returns a tensor of the same size as x, containing the values of d/dx sinc(x)
    def dsinc(x, return_sinc=False):
        with torch.enable_grad():
            x2 = x.clone().detach() # Not sure this .clone() is needed, but removing it saves little time anyway
            x2.requires_grad = True
            y = torch.sinc(x2)
            grad_sinc_out = torch.ones_like(y)
            dy = torch.autograd.grad(y, x2, grad_sinc_out, create_graph=False)[0]

        if return_sinc:
            return dy, y
        else:        
            return dy



#############################################################################################################
##                                            Segmented Cumsum                                             ##
#############################################################################################################

# This is the main function that calculates the segmented cumsum.
# Input arguments: 
#   max_seg_size: an upper bound on the maximal length of a contiguous segment in <segment_ids>. If not provided, detected automatically.
#   in_place:     if set to True, writes the output directly to <values> instead of allocating new memory.
#   thorough_verify_input: verifies the input for correctness. meant for debugging purposes. in particular, checks <segment_ids>
#                          for repeated ids of different segments, and looks for infs and nans in <values>.
#   always_use_pure_torch: when set to True, always uses the pure torch implementation.
#                          otherwise, when the input is on a cuda device, uses a custom cuda implementation.
#                          the cuda implementation has a better memory bottleneck.
#                          in terms of running time, both are comparable, with two-fold differences for one over
#                          the other or vice versa.
#
# Output: The segmented cumsum of <values> according to <segment_ids>.
def segcumsum(values, segment_ids, max_seg_size=None, in_place=False, thorough_verify_input=False, always_use_pure_torch=False):
    # Verify input device, dtypes and shapes
    assert values.dim() == 1, 'values must be a 1-dimensional tensor'
    assert segment_ids.dim() == 1, 'segment_ids must be a 1-dimensional tensor'
    assert segment_ids.numel() == values.numel(), 'values and segment_ids must contain the same number of elements'
    assert segment_ids.dtype in (torch.int32,torch.int64), 'segment_ids must have int32 or int64 dtype'
    assert values.device == segment_ids.device, 'values and segment_ids must be on the same device'

    # Ensure all data is contiguous
    assert not segment_ids.is_sparse, 'segment_ids cannot be sparse'
    assert segment_ids.is_contiguous(), 'segment_ids must be in contiguous format'

    assert not values.is_sparse, 'values cannot be sparse'
    assert (not in_place) or values.is_contiguous(), 'when in_place==True, values must be in contiguous format'

    num_segments = None

    if max_seg_size is None:
        # 0 seconds
        # Calculate maximal segmet size
        _, counts_consecutive = torch.unique_consecutive(segment_ids, return_counts=True)
        del _
        num_segments = counts_consecutive.numel()
        max_seg_size_real = int(torch.max(counts_consecutive))
        max_seg_size = max_seg_size_real
        del counts_consecutive
    else:
        assert isinstance(max_seg_size, numbers.Number)
        assert max_seg_size >= 1

    if thorough_verify_input:
        if num_segments is None:
            _, counts_consecutive = torch.unique_consecutive(segment_ids, return_counts=True)
            del _
            num_segments = counts_consecutive.numel()
            max_seg_size_real = int(torch.max(counts_consecutive))
            del counts_consecutive

        _, counts_total = torch.unique(segment_ids, return_counts=True)
        del _
        num_segments_unique = counts_total.numel()
        del counts_total

        assert num_segments == num_segments_unique, 'repeated segment IDs detected'
        assert max_seg_size == max_seg_size_real, 'incorrect max_seg_size detected (got %d, correct is %d)' % (max_seg_size, max_seg_size_real)

        assert not torch.isinf(values).any(), "Found infs in ''values''"
        assert not torch.isnan(values).any(), "Found nans in ''values''"

    # Calculate and return the segmented cumsum
    global libfsw_embedding

    if (values.device.type == 'cuda') and (not always_use_pure_torch) and (libfsw_embedding is not None):
        return segcumsum_cuda(values, segment_ids, max_seg_size, in_place)
    else:
        return segcumsum_torch(values, segment_ids, max_seg_size, in_place)
    
# torch implementation
def segcumsum_torch(values, segment_ids, max_seg_size, in_place):
    assert values.is_contiguous(), 'in the segcumsum_torch implementation, ''values'' must be in contiguous format'

    if in_place:
        out = values
    else:
        out = torch.clone(values, memory_format=torch.contiguous_format)

    return segcumsum_torch_main(out, segment_ids, max_seg_size)


# main loop of torch implementation
# Note: using torch jit here makes it empirically slower
def segcumsum_torch_main(values, segment_ids, max_seg_size : int):
    n = values.numel()

    stride = 0
    while stride < max_seg_size:
        stride = max(1, 2*stride)
        values[stride:n] += (segment_ids[stride:n] == segment_ids[0:(n-stride)]) * values[0:(n-stride)]
    
    return values


# cuda implementation
def segcumsum_cuda(values, segment_ids, max_seg_size, in_place):
    # Maximal number of CUDA threads to use per block.
    # Note: This is automatically capped by the maximal number supported by the architecture.
    # Set to an arbitrarily large number (e.g. 1e6) to determine automatically.
    max_num_threads_per_block = 1e6

    assert libfsw_embedding is not None, 'libfsw_embedding library is not loaded'

    assert values.device.type == 'cuda', 'the tensor ''values'' must be on a CUDA device'
    assert segment_ids.device.type == 'cuda', 'the tensor ''segment_ids'' must be on a CUDA device'
    assert segment_ids.dtype == torch.int64, 'segment_ids must have int64 dtype'
    
    # Process input data types
    if values.dtype == torch.float32:
        dtype_num = 0;
        c_num_type = ctypes.c_float
    elif values.dtype == torch.float64:
        dtype_num = 1;
        c_num_type = ctypes.c_double
    else:
        raise RuntimeError("Unsupported input_tensor dtype ''%s''" % (str(values.dtype)))

    n = values.numel()

    # TODO: Consider calculating this number at initialization. Calling the function get_max_threads_per_block takes ~0.7 seconds.
    # Determine the maximal number of threads per block supported in the current CUDA device
    # cuda_max_threads_per_block = get_max_threads_per_block(values.device.index)
    cuda_max_threads_per_block = 256

    # Take the smallest multiple of 32 greater or equal to the input size, but no less than 64
    num_threads_2 = max(64, (n+31)//32)

    threads_per_block = min(num_threads_2, max_num_threads_per_block, cuda_max_threads_per_block)
    shared_memory_size = threads_per_block * ctypes.sizeof(c_num_type)

    assert threads_per_block > 1, 'threads_per_block must be greater than 1'

    # Construct block hierarchy
    tensor_sizes = [ n, ]
    num_blocks = []
    max_seg_sizes = [ max_seg_size, ]
    
    # Stop dividing when the whole tensor fits in one block
    while tensor_sizes[-1] > threads_per_block:
        tensor_size_new = (tensor_sizes[-1] + threads_per_block - 1) // threads_per_block
        tensor_sizes.append(tensor_size_new)

        num_blocks.append(tensor_size_new)

        max_seg_size_new = (max_seg_sizes[-1] + threads_per_block - 1) // threads_per_block
        max_seg_sizes.append(max_seg_size_new)

    num_blocks.append(1)

    output_tensors = []
    id_tensors = []

    for i,s in enumerate(tensor_sizes):
        if i == 0:
            if in_place:
                output_tensor_new = values
            else:
                output_tensor_new = torch.clone(values, memory_format=torch.contiguous_format)

            id_tensor_new = segment_ids

        else:
            output_tensor_new = torch.empty(size=(s,), device=values.device, dtype=values.dtype, memory_format=torch.contiguous_format)
            id_tensor_new = torch.empty(size=(s,), device=segment_ids.device, dtype=segment_ids.dtype, memory_format=torch.contiguous_format)

        output_tensors.append(output_tensor_new)
        id_tensors.append(id_tensor_new)

    # Define the kernel signatures
    libfsw_embedding.segcumsum_wrapper.argtypes = [
        ctypes.c_int64,     # dtype_num
        ctypes.c_void_p,  # values input/output pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_int64,     # size
        ctypes.c_int64,     # max_seg_size
        ctypes.c_void_p,  # block sums output pointer
        ctypes.c_void_p,  # block last ids output pointer
        ctypes.c_bool,    # return_next_level boolean
        ctypes.c_int64,     # blocks
        ctypes.c_int64,     # threads_per_block
        ctypes.c_size_t   # shared_memory_size
    ]
    libfsw_embedding.segcumsum_wrapper.restype = None

    libfsw_embedding.add_block_sums_wrapper.argtypes = [
        ctypes.c_int64,     # dtype_num
        ctypes.c_void_p,  # output pointer
        ctypes.c_void_p,  # block_sums pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_void_p,  # block_last_id pointer
        ctypes.c_int64,     # size
        ctypes.c_int64,     # blocks
        ctypes.c_int64      # threads_per_block
    ]
    libfsw_embedding.add_block_sums_wrapper.restype = None

    for i,s in enumerate(tensor_sizes):
        return_next_level = ( i < (len(tensor_sizes) - 1) )

        # Launch the segcumsum_wrapper
        libfsw_embedding.segcumsum_wrapper(
            ctypes.c_int64(dtype_num),
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_int64(tensor_sizes[i]),            
            ctypes.c_int64(max_seg_sizes[i]),
            ctypes.c_void_p(output_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_void_p(id_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_bool( return_next_level  ),
            ctypes.c_int64(num_blocks[i]),
            ctypes.c_int64(threads_per_block),
            ctypes.c_size_t(shared_memory_size)
        )


    for i in reversed(range(len(tensor_sizes)-1)):

        # Launch the add_block_sums_wrapper
        libfsw_embedding.add_block_sums_wrapper(
            ctypes.c_int64(dtype_num),
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(output_tensors[i+1].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i+1].data_ptr()),
            ctypes.c_int64(tensor_sizes[i]),
            ctypes.c_int64(num_blocks[i]),
            ctypes.c_int64(threads_per_block)
        )

    return output_tensors[0]


# This is a slow alternative of segcumsum() to verify the correctness of the results
def segcumsum_slow(x, segment_ids):
    out = torch.empty_like(x)

    for i in range(len(x)):
        if (i == 0):
            out[i] = x[i]
        elif segment_ids[i] == segment_ids[i-1]:
            out[i] = out[i-1] + x[i]
        else:
            out[i] = x[i]
    
    return out


# Returns the maximal number of threads per block supported by the CUDA device with the given index
# Note: This function may take ~0.7 seconds to run
def get_max_threads_per_block(device_index):    
    libfsw_embedding.get_max_threads_per_block.argtypes = [ ctypes.c_int ]
    libfsw_embedding.get_max_threads_per_block.restype = ctypes.c_int
    out = libfsw_embedding.get_max_threads_per_block(ctypes.c_int(device_index))
    return out



#############################################################################################################
##                                      Mutual Coherence Minimization                                      ##
#############################################################################################################


def minimize_mutual_coherence(X_init, report=True):
    step_size_init = 2000
    nIter_max = 1000
    improvement_thresh = 1e-4 # Use 1e-6 for more thoroughness

    p_vals = [3,6,10,20,50,100,200,500,1000,2000,5000, 1e4, 2e4, 5e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]

    step_size_curr = step_size_init
    
    n = X_init.shape[0]
    X_curr = X_init
    X_curr = nn.functional.normalize(X_curr, p=2, dim=1, eps=0)
    mu_init = calc_mu_from_G(calc_G(X_curr))
    
    for ip, p_curr in enumerate(p_vals):
        qprintln(report, '\n=== Optimizing with p = %g (%d/%d) ===' % (p_curr, ip+1, len(p_vals)) )
        X_curr, step_size_curr = minimize_mutual_coherence_p(X_curr, p_curr, step_size_init=step_size_curr, improvement_thresh=improvement_thresh, nIter_max=nIter_max, report=report)

        mu_curr = mu_init = calc_mu_from_G(calc_G(X_curr))
        #qprintln('Relative improvement vs. init: %g' % ((mu_init-mu_curr)/(1-mu_init)))

        qprintln(report, '\nIncoherence: %g  Min. pairwise dist: %g' % ( 1.0-mu_curr, np.sqrt(np.maximum(0,2*(1.0-mu_curr.cpu()))) ) )
    return X_curr



def minimize_mutual_coherence_p(X_init, p, step_size_init, improvement_thresh, nIter_max, report):
    # Note: X_init must be normalized to unit rows

    p = np.double(p)

    # Parameters
    step_size_min = 1e-5
    step_size_max = 1e10
    max_num_low_improvements = 5
    step_decrease_factor = 0.5

    assert p >= 2, "p must be greater or equal to 2"
    assert step_size_init >= step_size_min, "Initial step size below minium"
    assert step_size_init <= step_size_max, "Initial step size above maximum"

    # Initialization
    n = X_init.shape[0]

    if X_init.numel() == 0:
        return X_init, step_size_init
    elif n == 1:
        return torch.nn.functional.normalize(X_init), step_size_init
    
    onevec = torch.ones([n,1], dtype=X_init.dtype, device=X_init.device)

    mu_init = calc_mu_from_G(calc_G(X_init))

    ## Initialize first iteration
    step_size_curr = step_size_init

    # The first step size is chosen as follows: Start from step_size_init.
    # If at the first iteration this step size yields an objective decrease,
    # iteratively increase the step size until the objective increases.
    # Then choose the best among the tested step sizes and use it. This
    # prevents getting stuck with too low a step size throughout the
    # optimization process.
    step_size_init_best = step_size_init
    obj_best_at_step_init_seek = np.inf
    finished_step_size_init = False

    low_improvement_counter = 0

    # These four have to be maintained together and consistent with each
    # other.
    X_curr = X_init
    G_curr = calc_G(X_curr) # The Gram matrix of X with its main diagonal annihilated.
    mu_curr, obj_curr = eval_G(G_curr, p)

    qprintln(report, '#%.2d:  Objective: %g  Surrogate incoherence: %g  Step size: %g' % (0, obj_curr, 1.0-obj_curr, step_size_curr) ) 

    rho = np.power( 1.0/(2.0*n*(n-1.0)), 1.0/p )

    for i in range(1, nIter_max+2):
        if i > nIter_max:
            qprintln(report, 'Reached maximal number of iterations. Breaking.')
            break

        # Calculate gradient at current solution
        # For numerical safety, G is normalized so that its
        # largest-magnitude entry equals 1.
        G_normalized = G_curr / mu_curr
        sum_offdiags_norm = torch.sum(torch.pow(torch.abs(G_normalized), p))
        
        grad_curr = rho / torch.pow(sum_offdiags_norm, 1.0 - 1.0/p) * ( ( torch.pow(torch.abs( G_normalized ), p-1.0) * torch.sign(G_normalized) ) @ X_curr - ( torch.pow(torch.abs(G_normalized), p) @ (mu_curr*onevec) ) * X_curr )

        # Calculate and evaluate new candidate solution
        X_new = nn.functional.normalize(X_curr - step_size_curr * grad_curr, p=2, dim=1, eps=0)
        G_new = calc_G(X_new)
        mu_new, obj_new = eval_G(G_new, p)

        # If the objective does not improve
        if obj_new >= obj_curr:
            if finished_step_size_init:
                # Decrease step size                
                if step_size_curr * step_decrease_factor < step_size_min:
                    qprintln(report, '#%.2d:  Objective does not improve at minimal step size. Breaking.' % (i) )
                    break

                step_size_curr = step_size_curr * step_decrease_factor
            else:
                # If we're still seeking the first step size, stop and pick
                # the best step size so far.
                step_size_curr = step_size_init_best
                finished_step_size_init = True
            
            qprintln(report, '#%.2d:  Decreaseing step size: %g' % (i, step_size_curr) )
            continue

        # --> If we're here, the objective has improved.
        
        if not finished_step_size_init:
            if (obj_new < obj_best_at_step_init_seek) and (step_size_curr / step_decrease_factor <= step_size_max):
                obj_best_at_step_init_seek = obj_new
                step_size_init_best = step_size_curr
                step_size_curr = step_size_curr / step_decrease_factor

                # Save the new best solution for later backtracking
                X_stepinit = X_new
                G_stepinit = G_new
                obj_stepinit = obj_new
                mu_stepinit = mu_new

                qprintln(report, '#%.2d:  Trying larger step size: %g' % (i, step_size_curr) )
                continue
            else:
                # If we're seeking the best first step, on the first time
                # that increasing the step does not improve the objective,
                # stop increasing and backtrack to the best solution so
                # far.
                step_size_curr = step_size_init_best
                finished_step_size_init = True

                # Backtrack to best candidate solution
                X_new = X_stepinit
                G_new = G_stepinit
                obj_new = obj_stepinit
                mu_new = mu_stepinit

        # --> If we're here, we accept the candidate solution.

        # We divide by 1-obj_curr rather than obj_curr because this measure
        # is more informative at values near 1.
        improvement_curr = (obj_curr-obj_new) / (1.0-obj_curr)

        # Accept candidate solution        
        X_curr = X_new
        G_curr = G_new
        obj_curr = obj_new
        mu_curr = mu_new
        
        qprint(report, '#%.2d:  Objective: %g  Surrogate incoherence: %g  Step size: %g  Improvement: %g  ' % (i, obj_curr, 1.0-obj_curr, step_size_curr, improvement_curr) )

        if improvement_curr <= improvement_thresh:
            low_improvement_counter =  low_improvement_counter + 1
            qprint(report, 'Low improvement strike %d' % (low_improvement_counter) )

            if low_improvement_counter >= max_num_low_improvements:
                qprintln(report)
                qprintln(report)
                qprintln(report, 'Reached maximal number of consecutive iterations with low improvement. Breaking.')
                break
        else:
            low_improvement_counter = 0

        qprintln(report)

    #qprintln(report, '\nIncoherence: %g' % (1-mu_curr) )

    if mu_curr < mu_init:
        qprintln(report, '\nRelative improvement with p=%g: %g' % (p, (mu_init-mu_curr)/(1.0-mu_init)) )
        X_out = X_curr
        step_size_out = step_size_curr
    else:
        qprintln(report, '\nIncoherence did not improve with p=%g. Reverting to previous solution.' % (p) )
        X_out = X_init
        step_size_out = step_size_init

    return X_out, step_size_out


def calc_G(X):    
    n = X.shape[0]
    G = X@X.transpose(0,1)
    G[range(n),range(n)] = 0
    return G


def calc_mu_from_G(G):
    return torch.max( torch.abs(G)  )


def eval_G(G, p):
    n = G.shape[0]
    mu = calc_mu_from_G(G)
    rho = 1.0/(2.0*n*(n-1.0))
    objective = mu * torch.pow( rho * torch.sum( torch.pow( torch.abs(G/mu), p ) ), 1.0/p )

    return mu, objective
