#ifndef TRAIN_CLI_HPP
#define TRAIN_CLI_HPP

#include "mlp.hpp"

#include <string>

namespace train_cli {

struct Options {
    std::string train_path;
    std::string valid_path;
    std::string scaler_path;
    std::string model_path;
    mlp::MLPConfig cfg;
    bool do_plot;
};

void print_usage(const char* prog);
bool parse_args(int argc, char** argv, Options& out, std::string& error);

} // namespace train_cli

#endif
