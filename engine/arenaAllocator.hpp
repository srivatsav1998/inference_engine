
#pragma once

#include <memory>
#include <iostream>

class ArenaAllocator
{
    size_t totalBytes_;
    void *head_;
    void *ptr_;
    size_t bytesAllocated_;

public:
    ArenaAllocator(size_t totalBytesToAllocate)
        : totalBytes_(totalBytesToAllocate),
          head_(malloc(totalBytesToAllocate)),
          ptr_(head_),
          bytesAllocated_(0)
    {
        if (head_ == nullptr)
        {
            std::cout << "Couldn't allocate requested amount of memory" << std::endl;
            throw std::runtime_error("Couldn't allocate requested amount of memory");
        }
    }

    float *alloc(size_t sizeInBytes)
    {
        // SIMD registers prefer a memory location that is a multiple of 16
        size_t roundOff = sizeInBytes % 16;
        if (roundOff != 0)
        {
            sizeInBytes += 16 - roundOff;
        }

        // check if allocation is even possible
        if (bytesAllocated_ + sizeInBytes > totalBytes_)
        {
            std::cout << "Out of memory" << std::endl;
            return nullptr;
        }

        // Allocation is possible
        float *retVal = reinterpret_cast<float *>(ptr_);
        ptr_ = reinterpret_cast<char *>(ptr_) + sizeInBytes;
        bytesAllocated_ += sizeInBytes;
        return retVal;
    }

    void reset()
    {
        bytesAllocated_ = 0;
        ptr_ = head_;
    }

    ~ArenaAllocator()
    {
        free(head_);
    }
};