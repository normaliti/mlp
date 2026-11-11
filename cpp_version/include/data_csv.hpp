#ifndef DATA_CSV_HPP
#define DATA_CSV_HPP

#include <string>
#include <vector>

namespace data {

struct CsvData {
    std::vector< std::vector<std::string> > rows;
};

CsvData read_csv(const std::string& path);
bool write_csv(const std::string& path, const CsvData& data);

} // namespace data

#endif
