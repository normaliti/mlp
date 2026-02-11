#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <vector>

namespace dataset {

struct Dataset {
    std::vector< std::vector<double> > x;
    std::vector<int> y;
};

struct Scaler {
    std::vector<double> min;
    std::vector<double> max;
};

Dataset load_dataset(const std::string& csv_path);
Scaler fit_minmax(const Dataset& data);
void apply_minmax(Dataset& data, const Scaler& scaler);
bool save_scaler(const std::string& path, const Scaler& scaler);
bool load_scaler(const std::string& path, Scaler& scaler);

} // namespace dataset

#endif
