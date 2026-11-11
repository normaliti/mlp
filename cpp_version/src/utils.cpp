#include "utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace utils {

void print_message(const std::string& name) {
    std::cout << "[mlp] " << name << std::endl;
}

int run_python_plot(const std::string& script, const std::string& arg) {
    std::error_code ec;
    std::filesystem::create_directories("models/.mplconfig", ec);
    if (ec) {
        std::cout << "Warning: could not create models/.mplconfig directory.\n";
    }

    std::string cmd = "MPLBACKEND=Agg MPLCONFIGDIR=models/.mplconfig python3 " + script;
    if (!arg.empty()) {
        cmd += " " + arg;
    }
    return std::system(cmd.c_str());
}

} // namespace utils
