#include "data_csv.hpp"

#include <fstream>
#include <sstream>

namespace data {

CsvData read_csv(const std::string& path) {
    CsvData out;
    std::ifstream file(path.c_str());
    if (!file) {
        return out;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        if (!row.empty()) {
            out.rows.push_back(row);
        }
    }
    return out;
}

bool write_csv(const std::string& path, const CsvData& data) {
    std::ofstream file(path.c_str());
    if (!file) {
        return false;
    }
    for (size_t i = 0; i < data.rows.size(); ++i) {
        const std::vector<std::string>& row = data.rows[i];
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j + 1 < row.size()) {
                file << ",";
            }
        }
        file << "\n";
    }
    return true;
}

} // namespace data
