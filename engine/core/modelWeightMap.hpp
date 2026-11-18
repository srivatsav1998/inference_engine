#pragma once
#include <sys/stat.h>

class ModelWeightMap
{
public:
    ~ModelWeightMap();

    // Will mount/ map the weights provided.
    // Will throw an exception if called after mounting once
    void mount(const char *filePath);

    // Idempotent call - Will unmount if something was mounted previously
    void unmount();

    // can return nullptr if nothing was mounted
    float *operator*() const;

private:
    int fd;
    float *file_memory = nullptr;
    struct stat sb;
};