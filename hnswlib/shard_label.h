#pragma once
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <functional>  // For std::hash
#include "hnswlib.h"


namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
class ShardedLabelLookup {
private:
    static constexpr size_t NUM_SHARDS = 128;  // Adjust based on expected concurrency
    struct Shard {
        std::unordered_map<labeltype, tableint> map;
        mutable std::shared_mutex mutex;  // Allows multiple readers or a single writer
    };
    std::vector<Shard> shards_;

public:
    ShardedLabelLookup() : shards_(NUM_SHARDS) {}

    // Get the shard index for a given label (using std::hash)
    size_t get_shard_index(labeltype label) const {
        return std::hash<labeltype>{}(label) % NUM_SHARDS;
    }

    // Thread-safe insert
    void insert(labeltype label, tableint id) {
        size_t idx = get_shard_index(label);
        std::unique_lock<std::shared_mutex> lock(shards_[idx].mutex);
        shards_[idx].map[label] = id;
    }

    // Thread-safe lookup (returns -1 if not found)
    tableint find(labeltype label) const {
        size_t idx = get_shard_index(label);
        std::shared_lock<std::shared_mutex> lock(shards_[idx].mutex);
        auto it = shards_[idx].map.find(label);
        return (it != shards_[idx].map.end()) ? it->second : -1;
    }

    // Thread-safe erase
    bool erase(labeltype label) {
        size_t idx = get_shard_index(label);
        std::unique_lock<std::shared_mutex> lock(shards_[idx].mutex);
        return shards_[idx].map.erase(label) > 0;
    }
};
} // namespace hnsw