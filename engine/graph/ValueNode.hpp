
#pragma once

#include "Node.hpp"
#include "tensorView.hpp"
#include "arenaAllocator.hpp"

using tensorview::types::TensorShape;

// A value node only contains the metadata required to be able to allocate memory when needed
// An operation which consumes this value and the operation that produces this will hold shared_ptrs
// to this node. This is how the lifetime of a value node is determined.
class ValueNode : public Node
{
public:
    // Used for model weights and other inputs
    ValueNode(TensorView &value) : value_(value) {};

    // Used for intermediate outputs
    ValueNode(size_t bytesRequired, TensorShape shape) : bytesRequired_(bytesRequired), shape_(shape) {}

    // Allocates and returns the value tensor if not already done.
    // If value node is already allocated, then it will simply return the value tensor
    TensorView &getValue(ArenaAllocator &allocator)
    {
        if (!value_)
        {
            value_ = TensorView(allocator.alloc(bytesRequired_), shape_, 0);
        }
        return value_;
    }

private:
    size_t bytesRequired_;
    TensorShape shape_;
    TensorView value_;
};