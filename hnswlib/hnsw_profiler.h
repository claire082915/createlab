#pragma once
#include <chrono>
#include <atomic>
#include <thread>
#include <string>
#include <mutex>
#include <vector>
#include <iostream>
#include <unordered_map>

namespace hnswlib {

class HNSWLightProfiler {
public:
    struct Event {
        std::string tag;
        std::thread::id tid;
        uint64_t duration_us;
        Event(const std::string& t, std::thread::id id, uint64_t d) : tag(t), tid(id), duration_us(d) {}
    };

    static std::mutex events_mutex_;
    static std::vector<Event> events_;
    static std::unordered_map<std::thread::id, uint64_t> thread_start_times_;
    static std::unordered_map<std::thread::id, uint64_t> thread_end_times_;
    static thread_local std::vector<Event> thread_local_events_;

    class Timer {
        std::string tag_;
        uint64_t start_;
    public:
        Timer(const std::string& tag) : tag_(tag) {
            start_ = now();
        }
        ~Timer() {
            uint64_t end = now();
            uint64_t duration = end - start_;
            auto tid = std::this_thread::get_id();
            
            thread_local_events_.emplace_back(tag_, tid, duration);
        
            // Record start/end times
            {
                std::lock_guard<std::mutex> lock(events_mutex_);
                if (HNSWLightProfiler::thread_start_times_.find(tid) == HNSWLightProfiler::thread_start_times_.end()) {
                    HNSWLightProfiler::thread_start_times_[tid] = start_;
                }
                HNSWLightProfiler::thread_end_times_[tid] = end;
            }
        }
        
    private:
        static uint64_t now() {
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
        }
    };

    // Function to flush thread-local data to global storage
    static void flush_thread_local() {
        if (!thread_local_events_.empty()) {
            std::lock_guard<std::mutex> lock(events_mutex_);
            events_.insert(events_.end(), thread_local_events_.begin(), thread_local_events_.end());
            thread_local_events_.clear();
        }
    }

    static void export_to_csv(const std::string& filename) {
        flush_thread_local();
        std::lock_guard<std::mutex> lock(events_mutex_);
        
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << "\n";
            return;
        }
    
        // CSV Header
        outfile << "Tag,ThreadID,Calls,TotalTime_ms,AverageTime_us,MinTime_us,MaxTime_us\n";
    
        // Aggregate data by tag and thread
        std::unordered_map<std::string, std::unordered_map<std::thread::id, std::vector<uint64_t>>> thread_tag_times;
        for (const auto& e : events_) {
            thread_tag_times[e.tag][e.tid].push_back(e.duration_us);
        }
    
        // Write data rows
        for (const auto& [tag, thread_map] : thread_tag_times) {
            for (const auto& [tid, times] : thread_map) {
                uint64_t total = 0;
                uint64_t min_time = UINT64_MAX;
                uint64_t max_time = 0;
                for (auto t : times) {
                    total += t;
                    if (t > max_time) max_time = t;
                    if (t < min_time) min_time = t;
                }
                double avg = static_cast<double>(total) / times.size();
                outfile << "\"" << tag << "\","
                        << tid << ","
                        << times.size() << ","
                        << total / 1000.0 << ","
                        << avg << ","
                        << min_time << ","
                        << max_time << "\n";
            }
        }
    
        outfile.close();
        std::cout << "Profiler data exported to CSV: " << filename << "\n";
    }

    static void report() {
        // First flush current thread's data
        flush_thread_local();
        
        std::lock_guard<std::mutex> lock(events_mutex_);
        
        // Aggregate data by thread and tag
        std::unordered_map<std::string, std::unordered_map<std::thread::id, std::vector<uint64_t>>> thread_tag_times;
        for (const auto& e : events_) {
            thread_tag_times[e.tag][e.tid].push_back(e.duration_us);
        }
    
        std::cout << "\n=== Fine-Grained HNSW Timing (Per Thread) ===\n";
        
        // Create a summary by thread ID
        std::unordered_map<std::thread::id, uint64_t> thread_totals;
        
        for (const auto& [tag, thread_map] : thread_tag_times) {
            std::cout << "\n--- " << tag << " ---\n";
            for (const auto& [tid, times] : thread_map) {
                uint64_t total = 0;
                uint64_t min_time = UINT64_MAX;
                uint64_t max_time = 0;
                for (auto t : times) {
                    total += t;
                    if (t > max_time) max_time = t;
                    if (t < min_time) min_time = t;
                }
                thread_totals[tid] += total;
                
                double avg = static_cast<double>(total) / times.size();
                std::cout << "  Thread " << tid
                          << " | calls: " << times.size()
                          << " | total(ms): " << total / 1000.0
                          << " | avg(us): " << avg
                          << " | min(us): " << min_time
                          << " | max(us): " << max_time << "\n";
            }
        }
        
        // Summary by thread
        // std::cout << "\n=== Thread Summary ===\n";
        // for (const auto& [tid, total] : thread_totals) {
        //     std::cout << "Thread " << tid << " | total time(ms): " << total / 1000.0 << "\n";
        // }
        
        std::cout << "\nTotal threads used: " << thread_totals.size() << "\n";
    }
    
    static void clear() {
        std::lock_guard<std::mutex> lock(events_mutex_);
        events_.clear();
        thread_local_events_.clear();
        thread_start_times_.clear();
        thread_end_times_.clear();
    }
};

// Static member definitions
std::mutex HNSWLightProfiler::events_mutex_;
std::vector<HNSWLightProfiler::Event> HNSWLightProfiler::events_;
std::unordered_map<std::thread::id, uint64_t> HNSWLightProfiler::thread_start_times_;
std::unordered_map<std::thread::id, uint64_t> HNSWLightProfiler::thread_end_times_;
thread_local std::vector<hnswlib::HNSWLightProfiler::Event> hnswlib::HNSWLightProfiler::thread_local_events_;

}  // namespace hnswlib