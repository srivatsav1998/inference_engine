#pragma once

#include "tensorView.hpp"
#include "opType.hpp"
#include "mathOps.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cassert>

// ============================================================================
// Abstract Base Class: Operator (Archetype 2 Polymorphic Hierarchy)
// ============================================================================
class Operator
{
public:
    virtual ~Operator() = default;

    [[nodiscard]] virtual OpType type() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    // Executes the operator kernel reading from inputs and writing into output buffer
    virtual void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const = 0;
};

// ============================================================================
// Concrete Operators with Baked Invariant Attributes
// ============================================================================

class MatMulOp : public Operator
{
    bool transposeA_;
    bool transposeB_;

public:
    explicit MatMulOp(bool transposeA = false, bool transposeB = false)
        : transposeA_(transposeA), transposeB_(transposeB) {}

    [[nodiscard]] OpType type() const override { return OpType::MatMul; }
    [[nodiscard]] std::string name() const override { return "MatMul"; }
    [[nodiscard]] bool transposeA() const { return transposeA_; }
    [[nodiscard]] bool transposeB() const { return transposeB_; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 2 && "MatMulOp requires exactly 2 inputs (A, B)");
        matMul2D_out(*inputs[0], *inputs[1], output, transposeA_, transposeB_);
    }
};

class AddOp : public Operator
{
public:
    [[nodiscard]] OpType type() const override { return OpType::Add; }
    [[nodiscard]] std::string name() const override { return "Add"; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 2 && "AddOp requires exactly 2 inputs (A, B)");
        add2D_out(*inputs[0], *inputs[1], output);
    }
};

class BiasAddOp : public Operator
{
public:
    [[nodiscard]] OpType type() const override { return OpType::BiasAdd; }
    [[nodiscard]] std::string name() const override { return "BiasAdd"; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 2 && "BiasAddOp requires exactly 2 inputs (A, b)");
        add1DTo2D_out(*inputs[0], *inputs[1], output);
    }
};

class GeluOp : public Operator
{
public:
    [[nodiscard]] OpType type() const override { return OpType::GELU; }
    [[nodiscard]] std::string name() const override { return "GELU"; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 1 && "GeluOp requires exactly 1 input (A)");
        gelu_out(*inputs[0], output);
    }
};

class SoftmaxOp : public Operator
{
    int axis_;

public:
    explicit SoftmaxOp(int axis = -1) : axis_(axis) {}

    [[nodiscard]] OpType type() const override { return OpType::Softmax; }
    [[nodiscard]] std::string name() const override { return "Softmax"; }
    [[nodiscard]] int axis() const { return axis_; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 1 && "SoftmaxOp requires exactly 1 input (A)");
        softmax_out(*inputs[0], output);
    }
};

class LayerNormOp : public Operator
{
    float eps_;

public:
    explicit LayerNormOp(float eps = 1e-5f) : eps_(eps) {}

    [[nodiscard]] OpType type() const override { return OpType::LayerNorm; }
    [[nodiscard]] std::string name() const override { return "LayerNorm"; }
    [[nodiscard]] float eps() const { return eps_; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 3 && "LayerNormOp requires 3 inputs (X, weight, bias)");
        (void)eps_; // currently using default epsilon in kernel
        layerNorm_out(*inputs[0], *inputs[1], *inputs[2], output);
    }
};

class EleMulOp : public Operator
{
    float val_;

public:
    explicit EleMulOp(float val) : val_(val) {}

    [[nodiscard]] OpType type() const override { return OpType::EleMul; }
    [[nodiscard]] std::string name() const override { return "EleMul"; }
    [[nodiscard]] float val() const { return val_; }

    void execute(const std::vector<const TensorView *> &inputs, TensorView &output) const override
    {
        assert(inputs.size() == 1 && "EleMulOp requires exactly 1 input (X)");
        eleMul2D_out(*inputs[0], val_, output);
    }
};
