#include "dataset.hpp"
#include "data_csv.hpp"

#include <cstdlib>
#include <fstream>
#include <limits>

namespace dataset {

Dataset load_dataset(const std::string& csv_path) {
    Dataset out;
    data::CsvData raw = data::read_csv(csv_path);
    for (size_t i = 0; i < raw.rows.size(); ++i) {
        const std::vector<std::string>& row = raw.rows[i];
        if (row.size() < 3) {
            continue;
        }
        // Column 0: id (skip), Column 1: label (M/B), Columns 2.. end: features
        int label = 0;
        if (row[1] == "M") {
            label = 1;
        } else if (row[1] == "B") {
            label = 0;
        } else {
            continue;
        }

        std::vector<double> features;
        features.reserve(row.size() - 2);
        for (size_t j = 2; j < row.size(); ++j) {
            features.push_back(std::atof(row[j].c_str()));
        }

        out.x.push_back(features);
        out.y.push_back(label);
    }
    return out;
}

Scaler fit_minmax(const Dataset& data) {
    Scaler s;
    if (data.x.empty()) {
        return s;
    }
    size_t n_features = data.x[0].size();
    s.min.assign(n_features, std::numeric_limits<double>::infinity());
    s.max.assign(n_features, -std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < data.x.size(); ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            double v = data.x[i][j];
            if (v < s.min[j]) s.min[j] = v;
            if (v > s.max[j]) s.max[j] = v;
        }
    }
    return s;
}

void apply_minmax(Dataset& data, const Scaler& scaler) {
    if (data.x.empty()) {
        return;
    }
    size_t n_features = data.x[0].size();
    for (size_t i = 0; i < data.x.size(); ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            double minv = scaler.min[j];
            double maxv = scaler.max[j];
            double denom = maxv - minv;
            if (denom == 0.0) {
                data.x[i][j] = 0.0;
            } else {
                data.x[i][j] = (data.x[i][j] - minv) / denom;
            }
        }
    }
}

bool save_scaler(const std::string& path, const Scaler& scaler) {
    std::ofstream file(path.c_str());
    if (!file) {
        return false;
    }
    file << scaler.min.size() << "\n";
    for (size_t i = 0; i < scaler.min.size(); ++i) {
        file << scaler.min[i] << " " << scaler.max[i] << "\n";
    }
    return true;
}

bool load_scaler(const std::string& path, Scaler& scaler) {
    std::ifstream file(path.c_str());
    if (!file) {
        return false;
    }
    size_t n = 0;
    file >> n;
    scaler.min.assign(n, 0.0);
    scaler.max.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        file >> scaler.min[i] >> scaler.max[i];
    }
    return true;
}

} // namespace dataset
