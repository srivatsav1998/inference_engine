#pragma once

#include <queue>
#include <functional>
#include <thread>
#include <vector>
#include <condition_variable>
#include <atomic>

// This class should be a singleton class that supports task enqueues.
// This class should also provide a conditional variable to indicate empty task queue
class TaskQueue
{
public:
    static TaskQueue &getInstance();

    void enqueueTask(std::function<void()> action);

    ~TaskQueue();

    std::atomic<size_t> tasksCnt_;
    std::condition_variable tasksCntCv_;
    std::mutex tasksCntMutex_;

private:
    TaskQueue();

    std::vector<std::thread> threads_;

    std::mutex tasksMutex_;
    std::condition_variable tasksCv_;
    std::queue<std::function<void()>> tasks_;

    std::atomic<bool> stopThreads_;
};