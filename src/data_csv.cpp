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

} // namespace data
