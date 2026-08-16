
#include "tensorView.hpp"

TensorView::TensorView(float *data, std::vector<size_t> shape, size_t byteOffSet)
    : data_(data + (byteOffSet / sizeof(float))), shape_(std::move(shape)), strides_(shape_.size())
{
    // Numpy's toBytes() writes the values in row-major format and it will stack height in the positive Z axis
    // For a three dimension array, shape would be provided as [height, rows, cols]
    // To climb a unit of height, you will have to cross all the rows * cols elements.
    // To climb a unit of row, you will have to cross all the cols
    // To climb a unit of col, you just move by 1.
    // So, the strides would be [rows*cols, cols, 1]

    size_t current_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        strides_[i] = current_stride;
        current_stride *= shape_[i];
    }
}

TensorView::TensorView(float *data, std::vector<size_t> shape, std::vector<size_t> stride, size_t byteOffSet)
    : data_(data + (byteOffSet / sizeof(float))), shape_(std::move(shape)), strides_(std::move(stride)) {}

size_t TensorView::ndim(void) const
{
    return shape_.size();
}

const std::vector<size_t> &TensorView::shape(void) const
{
    return shape_;
}

float *TensorView::get()
{
    return data_;
}

TensorView TensorView::horSplit(size_t startCol, size_t endCol) const
{
    float *startPtr = data_ + startCol;
    return TensorView(startPtr, {shape_[0], endCol - startCol}, strides_, 0);
}

const std::vector<size_t> &TensorView::stride(void) const
{
    return strides_;
}