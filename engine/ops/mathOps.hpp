
#pragma once

class TensorView;
class ArenaAllocator;

//////////////// Accepts output Variants ////////////////

// Matrix Multiplication: O = A * B.
// CONTRACT: Output buffer O MUST NOT alias input buffers A or B.
void matMul2D_out(const TensorView &A, const TensorView &B, TensorView &O);

// Element-wise Matrix Addition: O = A + B.
// CONTRACT: Supports in-place aliasing (O.get() == A.get() or O.get() == B.get()) with identical strides.
void add2D_out(const TensorView &A, const TensorView &B, TensorView &O);

// 1D Row Vector Broadcasting Addition: O = A + b (b has shape [1, cols]).
// CONTRACT: Supports in-place aliasing (O.get() == A.get()) with identical strides.
void add1DTo2D_out(const TensorView &A, const TensorView &b, TensorView &O);

// GELU Activation: O = GELU(A).
// CONTRACT: Pointwise operation. Supports in-place aliasing (O.get() == A.get()) with identical strides.
void gelu_out(const TensorView &A, TensorView &O);

// Row-wise Softmax: O = Softmax(A).
// CONTRACT: Supports in-place aliasing (O.get() == A.get()) with identical strides.
void softmax_out(const TensorView &A, TensorView &O);

// Layer Normalization: O = LayerNorm(X, weight, bias).
// CONTRACT: Supports in-place aliasing (O.get() == X.get()) with identical strides.
void layerNorm_out(const TensorView &X, const TensorView &weight, const TensorView &bias, TensorView &O);

// Scalar Multiplication: O = X * val.
// CONTRACT: Pointwise operation. Supports in-place aliasing (O.get() == X.get()) with identical strides.
void eleMul2D_out(const TensorView &X, float val, TensorView &O);

// This doesn't allocate memory but returns a new TensorView
TensorView transpose2D(TensorView &X);
