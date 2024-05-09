#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <nlohmann/json.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#ifndef TEXTIMAGEMATCHER_H
#define TEXTIMAGEMATCHER_H

// Usage:
// To use the singleton instance of TextImageMatcher, you would call:
// auto matcher = TextImageMatcher::getInstance("", 0.5f, 5)

class TextEmbeddingEntry {
public:
    std::string text;
    xt::xarray<double> embedding;  // Use xtensor array for embedding
    bool negative;
    bool ensemble;
    double probability;  // Use double for probability

    TextEmbeddingEntry(std::string txt, const std::vector<float>& emb, bool neg, bool ens)
        : text(txt), negative(neg), ensemble(ens), probability(0.0) {
        // Adapt the std::vector<float> to xt::xarray<double>
        embedding = xt::adapt(emb, {emb.size()});
    }
};

class Match {
    
public:
    int row_idx;
    std::string text;
    double similarity;  // Use double for similarity
    int entry_index;
    bool negative;
    bool passed_threshold;

    Match(int r_idx, std::string txt, double sim, int e_idx, bool neg, bool passed)
        : row_idx(r_idx), text(txt), similarity(sim), entry_index(e_idx), negative(neg), passed_threshold(passed) {}
};

class TextImageMatcher {
public:
    std::string model_name;
    double threshold;  // Use double for threshold
    int max_entries;
    bool run_softmax = true;
    std::vector<TextEmbeddingEntry> entries;
    std::string user_data = "";
    std::string text_prefix = "A photo of a ";

private:
    // Singleton instance
    static TextImageMatcher* instance;
    static std::mutex mutex;

    // Prevent Copy Construction and Assignment
    TextImageMatcher(const TextImageMatcher&) = delete;
    TextImageMatcher& operator=(const TextImageMatcher&) = delete;

    // Private Constructor
    TextImageMatcher(std::string m_name, double thresh, int max_ents)
        : model_name(m_name), threshold(thresh), max_entries(max_ents) {
        // Initialize entries with default TextEmbeddingEntry
        for (int i = 0; i < max_entries; ++i) {
            entries.push_back(TextEmbeddingEntry("", std::vector<float>(), false, false));
        }
    }
    std::atomic<bool> m_debug;//When set outputs all matches overrides match(report_all = false)

public:
    // Public Method to get the singleton instance
    static TextImageMatcher* getInstance(std::string model_name, float threshold, int max_entries) {
        std::lock_guard<std::mutex> lock(mutex); // Thread-safe in a multi-threaded environment
        if (instance == nullptr) {
            instance = new TextImageMatcher(model_name, threshold, max_entries);
        }
        return instance;
    }

    // Destructor
    ~TextImageMatcher() {
        // Cleanup code
    }

    void set_threshold(double new_threshold) {
        threshold = new_threshold;
    }

    void set_text_prefix(std::string new_text_prefix) {
        text_prefix = new_text_prefix;
    }

    std::vector<int> get_embeddings() {
        std::vector<int> valid_entries;
        for (size_t i = 0; i < entries.size(); i++) {
            if (!entries[i].text.empty()) {
                valid_entries.push_back(i);
            }
        }
        return valid_entries;
    }

    void load_embeddings(std::string filename) {
        if (!std::filesystem::exists(filename)) {
            std::ofstream file(filename);
            file.close();
            std::cout << "File " << filename << " does not exist, creating it." << std::endl;
        } else {
            try {
                std::ifstream f(filename);
                nlohmann::json data;
                f >> data;

                threshold = data["threshold"].get<double>();
                text_prefix = data["text_prefix"].get<std::string>();

                entries.clear();
                for (size_t i = 0; i < data["entries"].size(); i++) {
                    std::string text = data["entries"][i]["text"];
                    std::vector<float> embedding = data["entries"][i]["embedding"].get<std::vector<float>>();
                    bool negative = data["entries"][i]["negative"];
                    bool ensemble = data["entries"][i]["ensemble"];
                    entries.push_back(TextEmbeddingEntry(text, embedding, negative, ensemble));
                }
            } catch (const std::exception& e) {
                std::cout << "Error while loading file " << filename << ": " << e.what() << ". Maybe you forgot to save your embeddings?" << std::endl;
            }
        }
    }
    void set_debug(bool debug) {
        m_debug.store(debug);
        std::cout << "Setting debug to: " << m_debug.load() << std::endl;
    }

    std::vector<Match> match(const xt::xarray<double>& image_embedding_np, bool report_all = false) {
        
        bool report_all_debug = report_all || m_debug.load();

        std::vector<Match> results;
        // Ensure the input is a 2D array
        xt::xarray<double> image_embedding = image_embedding_np;
        if (image_embedding.dimension() == 1) {
            image_embedding = image_embedding.reshape({1, -1});
        }
        // Getting valid entries
        std::vector<int> valid_entries = get_embeddings();
        if (valid_entries.empty()) {
            return results; // Return an empty list if no valid entries
        }
        
        std::vector<xt::xarray<double>> to_stack;
        to_stack.reserve(valid_entries.size()); // Reserve memory in advance

        xt::xarray<double> text_embeddings_np; // Declare text_embeddings_np outside the if block

        if (!valid_entries.empty()) {
            for (size_t entry_idx : valid_entries) {
                if (entry_idx < entries.size()) {
                    to_stack.push_back(entries[entry_idx].embedding);
                } 
            }

            if (!to_stack.empty()) {
                // Initialize text_embeddings_np with the correct shape
                text_embeddings_np.resize({to_stack.size(), to_stack.front().size()});
                for (size_t i = 0; i < to_stack.size(); ++i) {
                    xt::view(text_embeddings_np, i, xt::all()) = to_stack[i];
                }
            }
        }
        
        // Looping through each image embedding
        for (std::size_t row_idx = 0; row_idx < image_embedding.shape()[0]; ++row_idx) {
            auto image_embedding_1d = xt::view(image_embedding, row_idx);
            xt::xarray<double> dot_products = xt::linalg::dot(text_embeddings_np, image_embedding_1d);
            xt::xarray<double> similarities;

            if (run_softmax) {
                similarities = xt::exp(100 * dot_products);
                double sum = xt::sum(similarities)();
                if (sum != 0) {
                    similarities /= sum;
                } else {
                    similarities = xt::zeros<double>({dot_products.size()});
                }
            } else {
                // These values are based on statistics collected for the RN50x4 model
                similarities = (dot_products - 0.27) / (0.41 - 0.27);
                similarities = xt::clip(similarities, 0, 1);
            }

            int best_idx = xt::argmax(similarities)();
            double best_similarity = similarities[best_idx];

            // Updating probabilities in entries
            for (size_t i = 0; i < similarities.size(); i++) {
                entries[valid_entries[i]].probability = similarities[i];
            }

            // Creating a new match object
            Match new_match(row_idx,
                            entries[valid_entries[best_idx]].text,
                            best_similarity,
                            valid_entries[best_idx],
                            entries[valid_entries[best_idx]].negative,
                            best_similarity > threshold);

            // Filtering results based on conditions
            if (!report_all_debug && new_match.negative) {
                continue;
            }
            if (report_all_debug || new_match.passed_threshold) {
                results.push_back(new_match);
            }
        }

        // print results
        // std::cout << "Best match output: [";
        // for (const auto& match : results) {
        //     std::cout << match.text << ", ";
        // }
        // std::cout << "]" << std::endl;

        return results;
    }    
};

#endif // TEXTIMAGEMATCHER_H
