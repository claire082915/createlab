// Usage example with OpenMP
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include "hnswlib/hnswlib.h"
#include "hnswlib/hnsw_profiler.h"
#include "hnswlib/shard_label.h"

#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <zlib.h>
#include <memory>
#include <algorithm>
#include <queue>
#include <cmath>
#include <set>
#include <numeric>
#include <limits>


class GzFileWrapper {
    private:
        gzFile file;
        
    public:
        GzFileWrapper(const std::string& filename, const char* mode) {
            file = gzopen(filename.c_str(), mode);
            if (!file) {
                throw std::runtime_error("Cannot open gzip file: " + filename);
            }
        }
        
        ~GzFileWrapper() {
            if (file) {
                gzclose(file);
            }
        }
        
        gzFile get() const { return file; }
        bool good() const { return file != nullptr; }
        
        int read(void* buf, unsigned len) {
            return gzread(file, buf, len);
        }
        
        z_off_t seek(z_off_t offset, int whence) {
            return gzseek(file, offset, whence);
        }
        
        z_off_t tell() {
            return gztell(file);
        }
};

std::vector<std::vector<int>> brute_force_ground_truth(
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& database,
    size_t k)
{
    std::vector<std::vector<int>> gt;
    if (queries.empty() || database.empty()) return gt;

    const size_t dim = queries[0].size();
    gt.resize(queries.size());

    #pragma omp parallel for
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& q = queries[qi];
        if (q.size() != dim) { gt[qi].clear(); continue; }

        std::vector<std::pair<float,int>> dists;
        dists.reserve(database.size());

        for (size_t di = 0; di < database.size(); ++di) {
            const auto& v = database[di];
            if (v.size() != dim) continue;                 // skip missing/invalid rows

            // squared L2 is fine for ranking (no sqrt)
            float dist = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                float diff = q[j] - v[j];
                dist += diff * diff;
            }
            dists.emplace_back(dist, static_cast<int>(di)); // label == index into inserted_vectors
        }

        const size_t actual_k = std::min(k, dists.size());
        if (actual_k == 0) { gt[qi].clear(); continue; }

        std::partial_sort(dists.begin(), dists.begin() + actual_k, dists.end());
        gt[qi].resize(actual_k);
        for (size_t i = 0; i < actual_k; ++i) {
            gt[qi][i] = dists[i].second;                   // return labels
        }
    }
    return gt;
}

// Function to extract vectors from HNSW index for ground truth calculation
// std::vector<std::vector<float>> extract_vectors_from_index(
//     const hnswlib::HierarchicalNSW<float>& index, int dim) {
    
//     std::cout << "Extracting vectors from HNSW index for ground truth calculation..." << std::endl;
    
//     std::vector<std::vector<float>> extracted_vectors;
//     size_t max_elements = index.max_elements_;
    
//     for (size_t i = 0; i < max_elements; ++i) {
//         try {
//             // Check if element exists in the index
//             auto label_lookup = index.label_lookup_.find(i);
//             if (label_lookup != index.label_lookup_.end()) {
//                 size_t internal_id = label_lookup->second;
                
//                 // Get the data pointer for this internal ID
//                 void* data_ptr = index.getDataByInternalId(internal_id);
//                 if (data_ptr != nullptr) {
//                     float* vec_data = static_cast<float*>(data_ptr);
//                     std::vector<float> vector(vec_data, vec_data + dim);
//                     extracted_vectors.push_back(std::move(vector));
//                 }
//             }
//         } catch (const std::exception& e) {
//             // Element doesn't exist or error accessing it, skip
//             continue;
//         }
//     }
    
//     std::cout << "Extracted " << extracted_vectors.size() << " vectors from index" << std::endl;
//     return extracted_vectors;
// }

// Function to calculate recall
float calculate_recall(const std::vector<int>& hnsw_results, 
                      const std::vector<int>& ground_truth, 
                      int k) {
    std::set<int> gt_set(ground_truth.begin(), ground_truth.begin() + std::min(k, static_cast<int>(ground_truth.size())));
    int matches = 0;
    
    for (int i = 0; i < std::min(k, static_cast<int>(hnsw_results.size())); ++i) {
        if (gt_set.count(hnsw_results[i])) {
            matches++;
        }
    }
    
    return static_cast<float>(matches) / k;
}

// Function to load BIGANN base vectors from compressed .bvecs.gz files
std::vector<std::vector<float>> load_bigann_base_gz(const std::string& filename, int max_vectors = -1) {
    GzFileWrapper gz_file(filename, "rb");
    
    std::vector<std::vector<float>> data;
    int vectors_loaded = 0;
    
    std::cout << "Loading from compressed file: " << filename << std::endl;
    
    while (gz_file.good() && (max_vectors == -1 || vectors_loaded < max_vectors)) {
        // Read dimension (first 4 bytes of each vector)
        int dim;
        int bytes_read = gz_file.read(&dim, sizeof(int));
        if (bytes_read != sizeof(int)) break;
        
        // Read vector data (unsigned char values)
        std::vector<unsigned char> temp(dim);
        bytes_read = gz_file.read(temp.data(), dim * sizeof(unsigned char));
        if (bytes_read != dim * sizeof(unsigned char)) break;
        
        // Convert to float
        std::vector<float> vector(dim);
        for (int j = 0; j < dim; j++) {
            vector[j] = static_cast<float>(temp[j]);
        }
        
        data.push_back(std::move(vector));
        vectors_loaded++;
        
        // Progress indicator for large loads
        if (vectors_loaded % 10000 == 0) {
            std::cout << "Loaded " << vectors_loaded << " vectors..." << std::endl;
        }
    }
    
    std::cout << "Loaded " << vectors_loaded << " vectors of dimension " << (data.empty() ? 0 : data[0].size()) << std::endl;
    return data;
}



void test_batch_parallel_addpoint_bigann_gz(int num_threads, const std::string& base_file, 
        const std::string& query_file,
        int num_elements = 10000, int num_queries = 1000,
        int M = 32, int ef_construction = 400) {
    hnswlib::HNSWLightProfiler::clear();
    std::cout << "Running persistent thread pool addPoint test with compressed BIGANN data\n";

    // Load BIGANN data from compressed file
    std::cout << "Loading BIGANN base data from .gz file..." << std::endl;
    auto data_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> bigann_data = load_bigann_base_gz(base_file, num_elements);
    auto data_end = std::chrono::high_resolution_clock::now();
    auto data_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(data_end - data_start).count();
    std::cout << "Data loading time: " << data_load_time << " ms" << std::endl;

    if (bigann_data.empty()) {
        std::cerr << "No data loaded!" << std::endl;
        return;
    }

    int dim = bigann_data[0].size();
    int actual_num_elements = bigann_data.size();

    // Build HNSW index
    hnswlib::L2Space space(dim);
    auto start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float> index(&space, actual_num_elements * 4, M, ef_construction);

    std::atomic<int> global_label_counter(0);
    std::vector<std::thread> threads;

    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(std::thread([&]() {
            while (true) {
                int label = global_label_counter.fetch_add(1);
                if (label >= actual_num_elements) break;

                try {
                    index.addPoint(bigann_data[label].data(), label);
                } catch (...) { /* ignore errors */ }
            }
            hnswlib::HNSWLightProfiler::flush_thread_local();
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Persistent threads: " << num_threads 
            << ", Build time: " << build_time << " ms"
            << ", Elements: " << index.cur_element_count 
            << ", Dimension: " << dim << std::endl;

    // Load query vectors
    std::cout << "\nLoading query vectors..." << std::endl;
    std::vector<std::vector<float>> query_data = load_bigann_base_gz(query_file, num_queries);
    if (query_data.empty()) {
        std::cerr << "No query data loaded!" << std::endl;
        return;
    }

    // Calculate ground truth from inserted vectors
    std::cout << "\nCalculating ground truth..." << std::endl;
    std::vector<std::vector<int>> ground_truth = brute_force_ground_truth(query_data, bigann_data, 100);

    // Run HNSW queries and compute recalls
    std::cout << "\nTesting HNSW queries..." << std::endl;
    index.setEf(400);  // higher efSearch for better recall

    std::vector<float> recalls_at_1, recalls_at_10, recalls_at_100;
    auto query_start = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < query_data.size(); ++q) {
        auto result = index.searchKnn(query_data[q].data(), 100);
        std::vector<int> hnsw_results;
        while (!result.empty()) {
            hnsw_results.push_back(result.top().second);
            result.pop();
        }
        std::reverse(hnsw_results.begin(), hnsw_results.end());

        recalls_at_1.push_back(calculate_recall(hnsw_results, ground_truth[q], 1));
        recalls_at_10.push_back(calculate_recall(hnsw_results, ground_truth[q], 10));
        recalls_at_100.push_back(calculate_recall(hnsw_results, ground_truth[q], 100));
    }

    auto query_end = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count();

    float avg_recall_1 = std::accumulate(recalls_at_1.begin(), recalls_at_1.end(), 0.0f) / recalls_at_1.size();
    float avg_recall_10 = std::accumulate(recalls_at_10.begin(), recalls_at_10.end(), 0.0f) / recalls_at_10.size();
    float avg_recall_100 = std::accumulate(recalls_at_100.begin(), recalls_at_100.end(), 0.0f) / recalls_at_100.size();

    std::cout << "\n=== Results ===\n";
    std::cout << "Query time: " << query_duration << " ms\n";
    std::cout << "Average query time: " << static_cast<float>(query_duration) / query_data.size() << " ms/query\n";
    std::cout << "Recall@1: " << avg_recall_1 << "\n";
    std::cout << "Recall@10: " << avg_recall_10 << "\n";
    std::cout << "Recall@100: " << avg_recall_100 << "\n";

    hnswlib::HNSWLightProfiler::export_to_csv(std::to_string(num_threads) + "_bigann_gz_profiler_output.csv");
    std::cout << "----------------------------------------\n";
}


// Streaming version that reads directly from compressed file (memory efficient)
void test_batch_parallel_addpoint_bigann_gz_streaming(int num_threads, const std::string& base_file, 
                                                      const std::string& query_file,
                                                      int num_elements = 10000, int num_queries = 1000,
                                                      int M = 48, int ef_construction = 800) {
    hnswlib::HNSWLightProfiler::clear();
    std::cout << "Running streaming compressed BIGANN addPoint test\n";
    
    // Read first vector to get dimension
    GzFileWrapper gz_file(base_file, "rb");
    int dim;
    gz_file.read(&dim, sizeof(int));
    
    std::cout << "Vector dimension: " << dim << ", Target elements: " << num_elements << std::endl;
    
    hnswlib::L2Space space(dim);
    auto start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float> index(&space, num_elements * 4, M, ef_construction);
    
    std::atomic<int> global_label_counter(0);
    std::mutex data_mutex;
    
    // Pre-load data in chunks to avoid concurrent file access issues with gzip
    std::vector<std::vector<float>> data_chunk;
    const int chunk_size = 1000;
    std::atomic<bool> data_exhausted(false);

    std::vector<std::vector<float>> inserted_vectors; 
    inserted_vectors.reserve(num_elements);
    
    // Data loading thread
    std::thread data_loader([&]() {
        GzFileWrapper loader_file(base_file, "rb");
        int vectors_loaded = 0;
        
        // Skip the dimension at the beginning
        int file_dim;
        loader_file.read(&file_dim, sizeof(int));
        
        while (vectors_loaded < num_elements && loader_file.good()) {
            std::vector<std::vector<float>> current_chunk;
            
            // Load chunk
            for (int i = 0; i < chunk_size && vectors_loaded < num_elements; i++) {
                // Read per-vector dim (mandatory for .bvecs)
                int dim_i = 0;
                int br = loader_file.read(&dim_i, sizeof(int));
                if (br != sizeof(int)) break;

                if (dim_i != dim) {
                    // std::cerr << "Unexpected per-vector dim: " << dim_i
                            // << " (expected " << dim << "). Aborting.\n";
                    break;
                }

                // Read vector payload
                std::vector<unsigned char> temp(dim_i);
                br = loader_file.read(temp.data(), dim_i * sizeof(unsigned char));
                if (br != dim_i * (int)sizeof(unsigned char)) break;

                std::vector<float> vector(dim_i);
                for (int j = 0; j < dim_i; ++j) vector[j] = static_cast<float>(temp[j]);

                current_chunk.push_back(std::move(vector));
                ++vectors_loaded;
            }
            
            // Add chunk to main data
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                for (auto& vec : current_chunk) {
                    data_chunk.push_back(std::move(vec));
                }
            }
            
            if (vectors_loaded > 0 && vectors_loaded % 10000 == 0) {
                std::cout << "Loaded " << vectors_loaded << " vectors..." << std::endl;
            }
        }
        
        data_exhausted = true;
        std::cout << "Data loading complete: " << vectors_loaded << " vectors" << std::endl;
    });
    
    // Wait for some initial data
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::vector<std::thread> threads;
    
    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(std::thread([&, thread_id] {
            while (true) {
                int label = global_label_counter.fetch_add(1);
                if (label >= num_elements) break;
                
                // Wait for data to be available
                std::vector<float> vector;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        if (label < data_chunk.size()) {
                            vector = data_chunk[label];
                            break;
                        }
                    }
                    
                    if (data_exhausted && label >= data_chunk.size()) {
                        break;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                if (vector.empty()) break;

                try {
                    index.addPoint(vector.data(), label);
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        if ((int)inserted_vectors.size() <= label) {
                            inserted_vectors.resize(label + 1);
                        }
                        inserted_vectors[label] = vector;
                    }
                } catch (const std::exception& e) {
                    // std::cerr << "Error adding point " << label << ": " << e.what() << std::endl;
                }
            }
            hnswlib::HNSWLightProfiler::flush_thread_local();
        }));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    data_loader.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Streaming compressed mode - Threads: " << num_threads 
              << ", Build time: " << duration << " ms"
              << ", Elements: " << index.cur_element_count 
              << ", Dimension: " << dim << std::endl;
    
    // Load query vectors and calculate ground truth (similar to non-streaming version)
    std::cout << "\nLoading query vectors..." << std::endl;
    std::vector<std::vector<float>> query_data = load_bigann_base_gz(query_file, num_queries);
    
    if (!query_data.empty()) {
        // Calculate ground truth using brute force from vectors actually in the HNSW index
        std::cout << "\nCalculating ground truth from HNSW index vectors..." << std::endl;
        std::vector<std::vector<int>> ground_truth = brute_force_ground_truth(query_data, inserted_vectors, 100);
        
        // Test HNSW queries and calculate recall
        std::cout << "\nTesting HNSW queries..." << std::endl;
        index.setEf(1000);
        
        std::vector<float> recalls_at_1, recalls_at_10, recalls_at_100;
        auto query_start = std::chrono::high_resolution_clock::now();
        
        for (size_t q = 0; q < query_data.size(); ++q) {
            auto result = index.searchKnn(query_data[q].data(), 100);
            
            std::vector<int> hnsw_results;
            while (!result.empty()) {
                hnsw_results.push_back(result.top().second);
                result.pop();
            }
            std::reverse(hnsw_results.begin(), hnsw_results.end());
            
            recalls_at_1.push_back(calculate_recall(hnsw_results, ground_truth[q], 1));
            recalls_at_10.push_back(calculate_recall(hnsw_results, ground_truth[q], 10));
            recalls_at_100.push_back(calculate_recall(hnsw_results, ground_truth[q], 100));
        }
        
        auto query_end = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count();
        
        float avg_recall_1 = std::accumulate(recalls_at_1.begin(), recalls_at_1.end(), 0.0f) / recalls_at_1.size();
        float avg_recall_10 = std::accumulate(recalls_at_10.begin(), recalls_at_10.end(), 0.0f) / recalls_at_10.size();
        float avg_recall_100 = std::accumulate(recalls_at_100.begin(), recalls_at_100.end(), 0.0f) / recalls_at_100.size();
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Query time: " << query_duration << " ms" << std::endl;
        std::cout << "Average query time: " << static_cast<float>(query_duration) / query_data.size() << " ms/query" << std::endl;
        std::cout << "Recall@1: " << avg_recall_1 << std::endl;
        std::cout << "Recall@10: " << avg_recall_10 << std::endl;
        std::cout << "Recall@100: " << avg_recall_100 << std::endl;
    }
              
    // hnswlib::HNSWLightProfiler::export_to_csv(std::to_string(num_threads) + "_bigann_gz_streaming_profiler_output.csv");
    std::cout << "----------------------------------------\n";
}

int main() {

    std::string base_file = "bigann_base.bvecs.gz";
    std::string query_file = "bigann_query.bvecs.gz";
    std::string learn_file = "bigann_learn.bvecs.gz";
    
    // Test with different thread counts
    std::vector<int> thread_counts = {32, 64};
    
    for (int threads : thread_counts) {
        std::cout << "\n=== Testing with " << threads << " threads ===\n";
        try {
            // Using streaming version with ground truth calculation
            // test_batch_parallel_addpoint_bigann_gz(threads, base_file, query_file, 100000, 100);
            test_batch_parallel_addpoint_bigann_gz_streaming(threads, base_file, query_file, 100000, 1000);
        } catch (const std::exception& e) {
            std::cerr << "Exception with " << threads << " threads: " << e.what() << std::endl;
            break;
        }
    }
    
    return 0;
}