#include "dataset.hpp"
#include "data_csv.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace dataset {

namespace {

std::string trim_ascii_ws(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }

    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(begin, end - begin);
}

bool parse_double_strict(const std::string& s, double& out) {
    const std::string trimmed = trim_ascii_ws(s);
    if (trimmed.empty()) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

} // namespace

Dataset load_dataset(const std::string& csv_path) {
    Dataset out;
    const data::CsvData raw = data::read_csv(csv_path);

    size_t expected_features = 0;
    for (size_t i = 0; i < raw.rows.size(); ++i) {
        const std::vector<std::string>& row = raw.rows[i];
        if (row.size() < 3) {
            continue;
        }

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
        bool row_ok = true;
        for (size_t j = 2; j < row.size(); ++j) {
            double parsed = 0.0;
            if (!parse_double_strict(row[j], parsed)) {
                row_ok = false;
                break;
            }
            features.push_back(parsed);
        }

        if (!row_ok || features.empty()) {
            continue;
        }

        if (expected_features == 0) {
            expected_features = features.size();
        }
        if (features.size() != expected_features) {
            continue;
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

    const size_t n_features = data.x[0].size();
    s.min = data.x[0];
    s.max = data.x[0];

    for (size_t i = 1; i < data.x.size(); ++i) {
        if (data.x[i].size() != n_features) {
            return Scaler();
        }
        for (size_t j = 0; j < n_features; ++j) {
            s.min[j] = std::min(s.min[j], data.x[i][j]);
            s.max[j] = std::max(s.max[j], data.x[i][j]);
        }
    }
    return s;
}

bool apply_minmax(Dataset& data, const Scaler& scaler) {
    if (data.x.empty()) {
        return true;
    }

    const size_t n_features = data.x[0].size();
    if (scaler.min.size() != n_features || scaler.max.size() != n_features) {
        return false;
    }

    for (size_t i = 0; i < data.x.size(); ++i) {
        if (data.x[i].size() != n_features) {
            return false;
        }
        for (size_t j = 0; j < n_features; ++j) {
            const double minv = scaler.min[j];
            const double maxv = scaler.max[j];
            const double denom = maxv - minv;
            if (denom == 0.0) {
                data.x[i][j] = 0.0;
            } else {
                data.x[i][j] = (data.x[i][j] - minv) / denom;
            }
        }
    }
    return true;
}

bool save_scaler(const std::string& path, const Scaler& scaler) {
    if (scaler.min.empty() || scaler.min.size() != scaler.max.size()) {
        return false;
    }

    std::ofstream file(path.c_str());
    if (!file) {
        return false;
    }

    file << scaler.min.size() << "\n";
    for (size_t i = 0; i < scaler.min.size(); ++i) {
        file << scaler.min[i] << " " << scaler.max[i] << "\n";
    }
    return static_cast<bool>(file);
}

bool load_scaler(const std::string& path, Scaler& scaler) {
    std::ifstream file(path.c_str());
    if (!file) {
        return false;
    }

    size_t n = 0;
    if (!(file >> n) || n == 0) {
        return false;
    }

    scaler.min.assign(n, 0.0);
    scaler.max.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (!(file >> scaler.min[i] >> scaler.max[i])) {
            scaler.min.clear();
            scaler.max.clear();
            return false;
        }
    }
    return true;
}

} // namespace dataset
