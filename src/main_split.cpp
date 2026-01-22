#include "utils.hpp"
#include "data_csv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void print_info(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <input_csv> <train_csv> <valid_csv> [seed] [valid_ratio]\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " data.csv data_training.csv data_validation.csv 42 0.2\n";
}

int main(int argc, char** argv) {
    utils::print_message("split");

    if (argc < 4) {
        print_info(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string train_path = argv[2];
    std::string valid_path = argv[3];

    unsigned int seed = 42;
    double valid_ratio = 0.2;

    if (argc >= 5) {
        seed = static_cast<unsigned int>(std::atoi(argv[4]));
    }
    if (argc >= 6) {
        valid_ratio = std::atof(argv[5]);
    }

    if (valid_ratio <= 0.0 || valid_ratio >= 1.0) {
        std::cout << "Error: valid_ratio must be between 0 and 1.\n";
        return 1;
    }

    data::CsvData all = data::read_csv(input_path);
    if (all.rows.empty()) {
        std::cout << "Error: failed to read input CSV or file is empty.\n";
        return 1;
    }

    std::vector< std::vector<std::string> > rows_m;
    std::vector< std::vector<std::string> > rows_b;

    for (size_t i = 0; i < all.rows.size(); ++i) {
        if (all.rows[i].size() < 2) {
            continue;
        }
        const std::string& label = all.rows[i][1];
        if (label == "M") {
            rows_m.push_back(all.rows[i]);
        } else if (label == "B") {
            rows_b.push_back(all.rows[i]);
        }
    }

    if (rows_m.empty() || rows_b.empty()) {
        std::cout << "Error: could not find both classes (M and B).\n";
        return 1;
    }

    std::mt19937 rng(seed);
    std::shuffle(rows_m.begin(), rows_m.end(), rng);
    std::shuffle(rows_b.begin(), rows_b.end(), rng);

    size_t valid_m = static_cast<size_t>(rows_m.size() * valid_ratio);
    size_t valid_b = static_cast<size_t>(rows_b.size() * valid_ratio);

    data::CsvData train_data;
    data::CsvData valid_data;

    for (size_t i = 0; i < rows_m.size(); ++i) {
        if (i < valid_m) {
            valid_data.rows.push_back(rows_m[i]);
        } else {
            train_data.rows.push_back(rows_m[i]);
        }
    }
    for (size_t i = 0; i < rows_b.size(); ++i) {
        if (i < valid_b) {
            valid_data.rows.push_back(rows_b[i]);
        } else {
            train_data.rows.push_back(rows_b[i]);
        }
    }

    std::shuffle(train_data.rows.begin(), train_data.rows.end(), rng);
    std::shuffle(valid_data.rows.begin(), valid_data.rows.end(), rng);

    if (!data::write_csv(train_path, train_data)) {
        std::cout << "Error: failed to write train CSV.\n";
        return 1;
    }
    if (!data::write_csv(valid_path, valid_data)) {
        std::cout << "Error: failed to write valid CSV.\n";
        return 1;
    }

    size_t total_m = rows_m.size();
    size_t total_b = rows_b.size();
    size_t train_m = total_m - valid_m;
    size_t train_b = total_b - valid_b;
    size_t valid_m_count = valid_m;
    size_t valid_b_count = valid_b;

    double total = static_cast<double>(total_m + total_b);
    double train_total = static_cast<double>(train_m + train_b);
    double valid_total = static_cast<double>(valid_m_count + valid_b_count);

    std::cout << "Total: " << (total_m + total_b) << "\n";
    std::cout << "Train: " << train_data.rows.size() << "\n";
    std::cout << "Valid: " << valid_data.rows.size() << "\n";
    std::cout << "Seed: " << seed << ", valid_ratio: " << valid_ratio << "\n";
    std::cout << "\nClass distribution:\n";
    std::cout << "  Total  - M: " << total_m << " (" << (total_m * 100.0 / total) << "%), "
              << "B: " << total_b << " (" << (total_b * 100.0 / total) << "%)\n";
    std::cout << "  Train  - M: " << train_m << " (" << (train_m * 100.0 / train_total) << "%), "
              << "B: " << train_b << " (" << (train_b * 100.0 / train_total) << "%)\n";
    std::cout << "  Valid  - M: " << valid_m_count << " (" << (valid_m_count * 100.0 / valid_total) << "%), "
              << "B: " << valid_b_count << " (" << (valid_b_count * 100.0 / valid_total) << "%)\n";
    return 0;
}
