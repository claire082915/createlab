// profiler.h
#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <mutex>

class Profiler {
public:
    struct FunctionStats {
        std::vector<long long> durations;
        long long total = 0;
    };

    struct LockStats {
        std::vector<long long> durations;
        long long total = 0;
    };

    // Static methods
    static void startTimer(const std::string& funcName);
    static void endTimer(const std::string& funcName);
    static void recordLockTime(const std::string& name, long long duration);
    static void recordTime(const std::string& name,
                         std::chrono::time_point<std::chrono::high_resolution_clock> start,
                         std::chrono::time_point<std::chrono::high_resolution_clock> end);
    static void printStats();
    static void printLockStats();
    static void printDetailedStats();

private:
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimes;
    static std::unordered_map<std::string, FunctionStats> functionStats;
    static std::unordered_map<std::string, LockStats> lockStats;
    static std::mutex lockStatsMutex;
    
    Profiler() = default;
    static Profiler& getInstance();
};

class TimedLock {
public:
    TimedLock(const std::string& lock_name, std::mutex& m);
    ~TimedLock();

private:
    std::string name;
    std::unique_lock<std::mutex> lock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#endif // PROFILER_H