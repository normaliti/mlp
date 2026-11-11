#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

namespace utils {

void print_message(const std::string& name);
int run_python_plot(const std::string& script, const std::string& arg);

} // namespace utils

#endif
