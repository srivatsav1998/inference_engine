
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

float *TensorView::get() const
{
    return data_;
}

TensorView TensorView::horSplit(size_t startCol, size_t endCol) const
{
    assert(shape_.size() == 2 && "horSplit is only defined for 2D views");
    assert(startCol < endCol && endCol <= shape_[1] && "horSplit needs 0 <= startCol < endCol <= cols");
    float *startPtr = data_ + strides_[1] * startCol;
    return TensorView(startPtr, {shape_[0], endCol - startCol}, strides_, 0);
}

TensorView TensorView::verSplit(size_t startRow, size_t endRow) const
{
    assert(shape_.size() == 2 && "verSplit is only defined for 2D views");
    assert(startRow < endRow && endRow <= shape_[0] && "verSplit needs 0 <= startRow < endRow <= rows");
    float *startPtr = data_ + strides_[0] * startRow;
    return TensorView(startPtr, {endRow - startRow, shape_[1]}, strides_, 0);
}

const std::vector<size_t> &TensorView::stride(void) const
{
    return strides_;
}