#include "taskQueue.hpp"

// initialize the thread pool and make them wait
TaskQueue::TaskQueue()
{
    // Initialize with zero tasks
    tasksCnt_.store(0);

    // Initialize stop threads with false
    stopThreads_.store(false);

    auto numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
    {
        // hardware_concurrency can sometimes return 0 when not implemented correctly.
        numThreads = 4; // fall back value
    }

    // reserving the correct size
    threads_.reserve(numThreads);

    for (size_t i = 0; i < numThreads; ++i)
    {
        auto action = [this]
        {
            while (true)
            {
                // all threads should forever conditionally wait for new tasks
                std::unique_lock<std::mutex> lck(this->tasksMutex_);
                this->tasksCv_.wait(lck, [this]
                                    { return (this->stopThreads_.load()) || (!this->tasks_.empty()); });

                // once awake, get the task from the tasks queue and act on it
                if (this->tasks_.empty() && this->stopThreads_.load())
                {
                    // call to terminate
                    return;
                };

                auto task = this->tasks_.front();
                this->tasks_.pop();

                lck.unlock();

                task();

                if (--this->tasksCnt_ == 0)
                {
                    this->tasksCntCv_.notify_all();
                }
            }
        };

        threads_.emplace_back([action]
                              { action(); });
    }
}

TaskQueue::~TaskQueue()
{
    stopThreads_.store(true);
    tasksCv_.notify_all();

    for (auto &thread : threads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

TaskQueue &TaskQueue::getInstance()
{
    static TaskQueue instance;
    return instance;
}

void TaskQueue::enqueueTask(std::function<void()> action)
{
    tasksCnt_++;
    {
        std::lock_guard lck(tasksMutex_);
        tasks_.push(std::move(action));
    }
    tasksCv_.notify_one();
}