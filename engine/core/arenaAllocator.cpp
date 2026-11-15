#include "arenaAllocator.hpp"

#include <cassert>
#include <cstdlib>
#include <new>

ArenaAllocator::ArenaAllocator(size_t totalBytesToAllocate)
    : totalBytes_(0), head_(nullptr), ptr_(nullptr), bytesAllocated_(0)
{
    reserve(totalBytesToAllocate);
}

size_t ArenaAllocator::alignedSize(size_t sizeInBytes)
{
    // SIMD registers prefer a memory location that is a multiple of 16
    return (sizeInBytes + 15) & ~static_cast<size_t>(15);
}

void ArenaAllocator::reserve(size_t bytes)
{
    assert(bytesAllocated_ == 0 && "ArenaAllocator::reserve called while allocations are live");
    if (bytes <= totalBytes_)
    {
        return;
    }

    void *block = malloc(bytes);
    if (block == nullptr)
    {
        throw std::bad_alloc();
    }

    free(head_);
    head_ = block;
    ptr_ = head_;
    totalBytes_ = bytes;
}

float *ArenaAllocator::alloc(size_t sizeInBytes)
{
    sizeInBytes = alignedSize(sizeInBytes);

    // check if allocation is even possible
    if (bytesAllocated_ + sizeInBytes > totalBytes_)
    {
        throw std::bad_alloc();
    }

    // Allocation is possible
    float *retVal = reinterpret_cast<float *>(ptr_);
    ptr_ = reinterpret_cast<char *>(ptr_) + sizeInBytes;
    bytesAllocated_ += sizeInBytes;
    return retVal;
}

void ArenaAllocator::reset()
{
    bytesAllocated_ = 0;
    ptr_ = head_;
}

size_t ArenaAllocator::bytesAllocated() const
{
    return bytesAllocated_;
}

size_t ArenaAllocator::capacity() const
{
    return totalBytes_;
}

ArenaAllocator::~ArenaAllocator()
{
    free(head_);
}
