"""
Mapping of PyTorch ATen operations to FLOPs per element.

Notes:
- FLOPs are counted per output element unless otherwise specified
- In-place operations (suffix '_') have same FLOP count as out-of-place versions
- Transcendental functions (exp, log, sin, etc.) are counted as 1 FLOP (convention)
- Actual hardware implementation may vary.
"""

# ==============================================================================
# BASIC ARITHMETIC OPERATIONS (1 FLOP per element)
# ==============================================================================

ATEN_FLOPS_PER_ELEMENT = {

    # Addition/Subtraction
    'aten::add': 1,
    'aten::add_': 1,
    'aten::sub': 1,
    'aten::sub_': 1,

    # Multiplication/Division
    'aten::mul': 1,
    'aten::mul_': 1,
    'aten::div': 1,
    'aten::div_': 1,
    'aten::true_divide': 1,
    'aten::floor_divide': 1,

    # Power operations
    'aten::pow': 1,  # Simplified; actual cost depends on exponent
    'aten::sqrt': 1,
    'aten::rsqrt': 1,  # Reciprocal square root
    'aten::square': 1,

    # Negation/Absolute
    'aten::neg': 1,
    'aten::neg_': 1,
    'aten::abs': 1,
    'aten::abs_': 1,

    # Reciprocal
    'aten::reciprocal': 1,
    'aten::reciprocal_': 1,

    # ==============================================================================
    # FUSED ARITHMETIC OPERATIONS (2-3 FLOPs per element)
    # ==============================================================================

    # Fused multiply-add variants
    'aten::addcmul': 3,      # input + value * (t1 * t2)
    'aten::addcmul_': 3,
    'aten::addcdiv': 3,      # input + value * (t1 / t2)
    'aten::addcdiv_': 3,
    'aten::lerp': 3,         # start + weight * (end - start)
    'aten::lerp_': 3,

    # Multiply-add
    'aten::addmm': None,     # Special: depends on matrix dims (see below)
    'aten::addmv': None,     # Special: depends on vector dims
    'aten::addr': None,      # Special: matrix += alpha * v1.outer(v2)

    # ==============================================================================
    # COMPARISON OPERATIONS (1 FLOP per element)
    # ==============================================================================

    'aten::eq': 1,
    'aten::ne': 1,
    'aten::lt': 1,
    'aten::le': 1,
    'aten::gt': 1,
    'aten::ge': 1,
    'aten::equal': 1,

    # ==============================================================================
    # LOGICAL OPERATIONS (counted as 1 FLOP per element by convention)
    # ==============================================================================

    'aten::logical_and': 1,
    'aten::logical_or': 1,
    'aten::logical_not': 1,
    'aten::logical_xor': 1,

    # ==============================================================================
    # TRANSCENDENTAL FUNCTIONS (counted as 1 FLOP per element by convention)
    # ==============================================================================

    # Exponential/Logarithmic
    'aten::exp': 1,
    'aten::exp_': 1,
    'aten::exp2': 1,
    'aten::expm1': 1,        # exp(x) - 1
    'aten::log': 1,
    'aten::log_': 1,
    'aten::log2': 1,
    'aten::log10': 1,
    'aten::log1p': 1,        # log(1 + x)

    # Trigonometric
    'aten::sin': 1,
    'aten::sin_': 1,
    'aten::cos': 1,
    'aten::cos_': 1,
    'aten::tan': 1,
    'aten::tan_': 1,
    'aten::asin': 1,
    'aten::acos': 1,
    'aten::atan': 1,
    'aten::atan2': 1,

    # Hyperbolic
    'aten::sinh': 1,
    'aten::cosh': 1,
    'aten::tanh': 1,
    'aten::tanh_': 1,
    'aten::asinh': 1,
    'aten::acosh': 1,
    'aten::atanh': 1,

    # Sigmoid/Logit
    'aten::sigmoid': 1,
    'aten::sigmoid_': 1,
    'aten::logit': 1,

    # ==============================================================================
    # ACTIVATION FUNCTIONS
    # ==============================================================================

    'aten::relu': 1,         # max(0, x)
    'aten::relu_': 1,
    'aten::gelu': 4,         # Gaussian Error Linear Unit (multiple ops)
    'aten::silu': 3,         # x * sigmoid(x)
    'aten::mish': 4,         # x * tanh(softplus(x))
    'aten::softplus': 2,     # log(1 + exp(x))
    'aten::elu': 2,          # alpha * (exp(x) - 1) for x < 0
    'aten::elu_': 2,
    'aten::selu': 2,
    'aten::leaky_relu': 1,   # max(x, alpha * x)
    'aten::leaky_relu_': 1,
    'aten::prelu': 2,        # Parametric ReLU
    'aten::rrelu': 1,        # Randomized ReLU
    'aten::hardtanh': 1,     # Clamp operation
    'aten::hardtanh_': 1,
    'aten::threshold': 1,
    'aten::threshold_': 1,

    # ==============================================================================
    # CLAMPING / ROUNDING
    # ==============================================================================

    'aten::clamp': 1,
    'aten::clamp_': 1,
    'aten::clamp_min': 1,
    'aten::clamp_max': 1,
    'aten::clip': 1,
    'aten::ceil': 1,
    'aten::floor': 1,
    'aten::round': 1,
    'aten::trunc': 1,
    'aten::frac': 1,         # Fractional part

    # ==============================================================================
    # REDUCTION OPERATIONS (N-1 FLOPs for N elements, or use special formula)
    # ==============================================================================

    'aten::sum': None,       # Special: N-1 additions for N elements
    'aten::mean': None,      # Special: sum + 1 division
    'aten::prod': None,      # Special: N-1 multiplications
    'aten::max': None,       # Special: N-1 comparisons
    'aten::min': None,       # Special: N-1 comparisons
    'aten::argmax': None,    # Special: N-1 comparisons
    'aten::argmin': None,    # Special: N-1 comparisons
    'aten::std': None,       # Special: complex computation
    'aten::var': None,       # Special: complex computation
    'aten::norm': None,      # Special: depends on norm type

    # ==============================================================================
    # MATRIX OPERATIONS (require dimension-specific calculations)
    # ==============================================================================

    'aten::mm': None,        # Matrix multiply: M*N*K for (M,K) @ (K,N)
    'aten::bmm': None,       # Batch matrix multiply
    'aten::matmul': None,    # General matrix multiply
    'aten::mv': None,        # Matrix-vector multiply: M*N for (M,N) @ (N,)
    'aten::dot': None,       # Dot product: 2N-1 for vectors of length N
    'aten::inner': None,     # Inner product
    'aten::outer': None,     # Outer product: M*N for (M,) x (N,)

    # Linear algebra
    'aten::linear': None,    # Special: matmul + optional bias add
    'aten::addmm': None,     # alpha * mat1 @ mat2 + beta * input

    # ==============================================================================
    # CONVOLUTION OPERATIONS (require dimension-specific calculations)
    # ==============================================================================

    'aten::conv1d': None,    # Special: complex formula
    'aten::conv2d': None,    # Special: Cout * Hout * Wout * (Cin * Kh * Kw)
    'aten::conv3d': None,    # Special: extend 2D formula
    'aten::conv_transpose1d': None,
    'aten::conv_transpose2d': None,
    'aten::conv_transpose3d': None,

    # ==============================================================================
    # NORMALIZATION (complex, multiple operations)
    # ==============================================================================

    'aten::batch_norm': None,      # Special: mean, var, normalize, scale, shift
    'aten::layer_norm': None,      # Special: similar to batch_norm
    'aten::group_norm': None,
    'aten::instance_norm': None,
    'aten::normalize': None,       # L2 normalization

    # ==============================================================================
    # SOFTMAX / LOG_SOFTMAX (complex operations)
    # ==============================================================================

    'aten::softmax': None,         # Special: exp + sum + div per element
    'aten::log_softmax': None,     # Special: similar to softmax
    'aten::_softmax': None,
    'aten::_log_softmax': None,

    # ==============================================================================
    # MEMORY OPERATIONS (0 FLOPs - no computation)
    # ==============================================================================

    'aten::copy_': 0,
    'aten::clone': 0,
    'aten::fill_': 0,
    'aten::zero_': 0,
    'aten::ones': 0,
    'aten::zeros': 0,
    'aten::empty': 0,
    'aten::empty_like': 0,
    'aten::zeros_like': 0,
    'aten::ones_like': 0,
    'aten::full': 0,
    'aten::full_like': 0,

    # Shape operations
    'aten::view': 0,
    'aten::reshape': 0,
    'aten::transpose': 0,
    'aten::permute': 0,
    'aten::squeeze': 0,
    'aten::unsqueeze': 0,
    'aten::flatten': 0,
    'aten::contiguous': 0,

    # Indexing/Slicing
    'aten::slice': 0,
    'aten::select': 0,
    'aten::index': 0,
    'aten::index_select': 0,
    'aten::masked_select': 0,
    'aten::gather': 0,
    'aten::scatter': 0,
    'aten::scatter_': 0,

    # Concatenation/Splitting
    'aten::cat': 0,
    'aten::stack': 0,
    'aten::split': 0,
    'aten::chunk': 0,

    # Type conversions
    'aten::to': 0,
    'aten::_to_copy': 0,
    'aten::type_as': 0,

    # Metadata
    'aten::size': 0,
    'aten::stride': 0,
    'aten::item': 0,
    'aten::detach': 0,
    'aten::detach_': 0,
}
