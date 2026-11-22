#pragma once

#include "Node.hpp"
#include "ValueNode.hpp"
#include "opType.hpp"

namespace operation::types
{
    using Values = std::vector<std::shared_ptr<ValueNode>>;
};

using operation::types::Values;

class OperationNode : public Node
{
public:
    OperationNode(Values &inputs, Values &outputs, OpType operation, OpAttrs attrs) : inputs_(inputs), outputs_(outputs), operation_(operation), attrs_(attrs) {};

private:
    Values inputs_;
    Values outputs_;
    OpType operation_;
    OpAttrs attrs_;
};