#pragma once

#include <vector>
#include <memory>
#include <cassert>

class TensorView
{
    float *data_;                 // true non-owning implementation
    std::vector<size_t> shape_;   // n-dimensional array
    std::vector<size_t> strides_; // should match the shape

public:
    TensorView() = default;

    TensorView(float *data, std::vector<size_t> shape, size_t byteOffSet);

    TensorView(float *data, std::vector<size_t> shape, std::vector<size_t> stride, size_t byteOffSet);

    template <typename... Args>
    [[nodiscard]] float &operator()(Args... indices)
    {
        static_assert(sizeof...(indices) > 0, "Must provide atleast one index");
        assert(sizeof...(indices) == shape_.size() && "Number of indices should match the tensor dimensions!");

        const size_t idx_array[] = {static_cast<size_t>(indices)...};

        size_t flat_offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            assert(idx_array[i] < shape_[i] && "Index out of bounds!");
            flat_offset += idx_array[i] * strides_[i];
        }

        return data_[flat_offset];
    }

    [[nodiscard]] size_t ndim(void) const;

    [[nodiscard]] const std::vector<size_t> &shape(void) const;

    [[nodiscard]] float *get();

    [[nodiscard]] const std::vector<size_t> &stride(void) const;

    // The endCol is exclusive
    [[nodiscard]] TensorView horSplit(size_t startCol, size_t endCol) const;
};