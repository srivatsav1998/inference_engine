#pragma once
#include <sys/stat.h>

class ModelWeightsLoader
{
public:
    ModelWeightsLoader(const char *filePath);
    ~ModelWeightsLoader();
    float *operator*() const;

private:
    int fd;
    float *file_memory = nullptr;
    struct stat sb;
};