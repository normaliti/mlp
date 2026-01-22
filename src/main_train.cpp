#include "utils.hpp"
#include "mlp.hpp"
#include <iostream>

int main() {
    utils::print_banner("train");
    mlp::MLP model;
    model.train();
    return 0;
}
