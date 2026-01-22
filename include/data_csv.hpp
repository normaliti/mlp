#ifndef DATA_CSV_HPP
#define DATA_CSV_HPP

#include <string>
#include <vector>

namespace data {

struct CsvData {
    std::vector< std::vector<std::string> > rows;
};

CsvData read_csv(const std::string& path);

} // namespace data

#endif
