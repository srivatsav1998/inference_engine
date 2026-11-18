#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

#include "modelWeightMap.hpp"

void ModelWeightMap::mount(const char *filePath)
{
    if (file_memory != nullptr)
    {
        throw std::logic_error("Cannot call mount on an already mounted weight map");
    }

    fd = open(filePath, O_RDONLY);
    // check for validity of file descriptor
    if (fd < 0)
    {
        throw std::runtime_error("Failed to open the file " + std::string(filePath));
    }

    if (fstat(fd, &sb) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to get file size");
    }

    if (sb.st_size == 0)
    {
        close(fd);
        throw std::logic_error("File is empty");
    }

    file_memory = static_cast<float *>(mmap(
        nullptr,     // Let OS choose address
        sb.st_size,  // Length of the mapping
        PROT_READ,   // Read-only access
        MAP_PRIVATE, // Private copy-on-write
        fd,          // File descriptor
        0            // Offset (must be page-aligned)
        ));

    if (file_memory == MAP_FAILED)
    {
        std::cerr << "Memory mapping failed.\n";
        close(fd);
        throw std::runtime_error("Memory mapping failed");
    }

    close(fd);
}

void ModelWeightMap::unmount()
{
    if (file_memory != nullptr)
    {
        if (munmap(file_memory, sb.st_size) == -1)
        {
            std::cerr << "Failed to unmap memory.\n";
            std::abort();
        }
    }
}

ModelWeightMap::~ModelWeightMap()
{
    unmount();
}

float *ModelWeightMap::operator*() const
{
    return file_memory;
}