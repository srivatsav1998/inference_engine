#pragma once

#include <string>

class Node
{
    std::string layerName_;

public:
    Node() {};
    Node(std::string layerName) : layerName_(layerName) {};
};