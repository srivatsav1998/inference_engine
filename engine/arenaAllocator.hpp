
#pragma once

#include <memory>

class ArenaAllocator
{
    size_t totalBytes_;
    void *head_;
    void *ptr_;
    size_t bytesAllocated_;

public:
    ArenaAllocator(size_t totalBytesToAllocate);

    float *alloc(size_t sizeInBytes);

    void reset();

    ~ArenaAllocator();
};