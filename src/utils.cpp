#include "utils.hpp"

#include <iostream>

namespace utils {

void print_banner(const std::string& name) {
    std::cout << "[mlp] " << name << std::endl;
}

} // namespace utils
