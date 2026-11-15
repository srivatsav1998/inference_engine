#pragma once

#include <cstddef>

class ArenaAllocator
{
    size_t totalBytes_;
    void *head_;
    void *ptr_;
    size_t bytesAllocated_;

public:
    explicit ArenaAllocator(size_t totalBytesToAllocate = 0);

    ArenaAllocator(const ArenaAllocator &) = delete;
    ArenaAllocator &operator=(const ArenaAllocator &) = delete;

    // Size an allocation actually occupies: rounded up to 16 bytes so every block is SIMD-aligned.
    // Exposed so callers can compute exact arena requirements ahead of time.
    [[nodiscard]] static size_t alignedSize(size_t sizeInBytes);

    // Grows capacity to at least `bytes` (never shrinks). Only valid while nothing is allocated,
    // i.e. right after construction or reset(). Throws std::bad_alloc if the OS refuses.
    void reserve(size_t bytes);

    // Throws std::bad_alloc when the arena does not have room: callers never see nullptr.
    float *alloc(size_t sizeInBytes);

    void reset();

    [[nodiscard]] size_t bytesAllocated() const;

    [[nodiscard]] size_t capacity() const;

    ~ArenaAllocator();
};
