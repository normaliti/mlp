#ifndef MLP_IO_HPP
#define MLP_IO_HPP

#include <string>

namespace mlp {

class MLP;

bool save_model(const MLP& model, const std::string& path);
bool load_model(MLP& model, const std::string& path);

} // namespace mlp

#endif
