#include "utils.hpp"
#include "mlp.hpp"
#include <iostream>

int main() {
    utils::print_banner("predict");
    mlp::MLP model;
    model.predict();
    return 0;
}
