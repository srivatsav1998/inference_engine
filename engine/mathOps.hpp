
#pragma once

class TensorView;
class ArenaAllocator;

//////////////// Accepts output Variants ////////////////

void matMul2D_out(TensorView &A, TensorView &B, TensorView &O);

//////////////// General Variants ////////////////

TensorView matMul2D(TensorView &A, TensorView &B, ArenaAllocator &alloc);

TensorView add2D(TensorView &A, TensorView &B, ArenaAllocator &alloc);

TensorView add1DTo2D(TensorView &A, TensorView &b, ArenaAllocator &alloc);

TensorView gelu(TensorView &A, ArenaAllocator &alloc);

TensorView softmax(TensorView &A, ArenaAllocator &alloc);

TensorView layerNorm(TensorView &X, TensorView &weight, TensorView &bias, ArenaAllocator &alloc);

// This doesn't allocate memory but returns a new TensorView
TensorView transpose2D(TensorView &X);

//////////////// Inplace Variants ////////////////

void add2D_inplace(TensorView &A, TensorView &B);

void add1DTo2D_inplace(TensorView &A, TensorView &b);

void gelu_inplace(TensorView &A);

void softmax_inplace(TensorView &A);

void eleMul2D_inplace(TensorView &X, float val);
