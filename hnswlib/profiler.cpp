// profiler.cpp
#include "profiler.h"
#include <mutex>

// Initialize static members
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> Profiler::startTimes;
std::unordered_map<std::string, Profiler::FunctionStats> Profiler::functionStats;
std::unordered_map<std::string, Profiler::LockStats> Profiler::lockStats;
std::mutex Profiler::lockStatsMutex;

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

// Implement all Profiler methods
void Profiler::startTimer(const std::string& funcName) {
    auto& profiler = getInstance();
    profiler.startTimes[funcName] = std::chrono::high_resolution_clock::now();
}

void Profiler::endTimer(const std::string& funcName) {
    auto& profiler = getInstance();
    auto end = std::chrono::high_resolution_clock::now();
    auto it = profiler.startTimes.find(funcName);
    
    if (it != profiler.startTimes.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count();
        profiler.functionStats[funcName].durations.push_back(duration);
        profiler.functionStats[funcName].total += duration;
        profiler.startTimes.erase(it);
    }
}

void Profiler::recordLockTime(const std::string& name, long long duration) {
    std::lock_guard<std::mutex> lock(lockStatsMutex);
    lockStats[name].durations.push_back(duration);
    lockStats[name].total += duration;
}

void Profiler::recordTime(const std::string& name,
                         std::chrono::time_point<std::chrono::high_resolution_clock> start,
                         std::chrono::time_point<std::chrono::high_resolution_clock> end) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto& profiler = getInstance();
    profiler.functionStats[name].durations.push_back(duration);
    profiler.functionStats[name].total += duration;
}

void Profiler::printStats() {
    auto& profiler = getInstance();
    std::cout << "\n=== Function Timing Statistics ===\n";
    std::cout << "Function Name\t\tCall Count\tTotal Time(ms)\tAvg Time(ms)\tMin Time(ms)\tMax Time(ms)\n";
    
    for (const auto& [funcName, stats] : profiler.functionStats) {
        if (stats.durations.empty()) continue;
        
        auto min = *std::min_element(stats.durations.begin(), stats.durations.end());
        auto max = *std::max_element(stats.durations.begin(), stats.durations.end());
        double avg = static_cast<double>(stats.total) / stats.durations.size();
        
        printf("%-20s\t%zu\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n", 
               funcName.c_str(), 
               stats.durations.size(),
               stats.total / 1000.0,
               avg / 1000.0,
               min / 1000.0,
               max / 1000.0);
    }
}

void Profiler::printLockStats() {
    std::lock_guard<std::mutex> lock(lockStatsMutex);
    std::cout << "\n=== Lock Timing Statistics ===\n";
    std::cout << "Lock Name\t\tCall Count\tTotal Time(ms)\tAvg Time(ms)\n";
    
    for (const auto& [name, stats] : lockStats) {
        if (stats.durations.empty()) continue;
        
        double avg = static_cast<double>(stats.total) / stats.durations.size();
        printf("%-20s\t%zu\t\t%.2f\t\t%.2f\n", 
               name.c_str(), 
               stats.durations.size(),
               stats.total / 1000.0,
               avg / 1000.0);
    }
}

void Profiler::printDetailedStats() {
    printStats();
    printLockStats();
}

// TimedLock implementations
TimedLock::TimedLock(const std::string& lock_name, std::mutex& m) 
    : name(lock_name), lock(m), start(std::chrono::high_resolution_clock::now()) {}

TimedLock::~TimedLock() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    Profiler::recordLockTime(name, duration);
}