#include "train_cli.hpp"

#include <cstdlib>
#include <iostream>

namespace train_cli {

void print_usage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <train_csv> <valid_csv> [options]\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog
              << " data_training.csv data_validation.csv"
              << " --layers 30 24 24 2 --epochs 50 --batch_size 8"
              << " --lr 0.1 --activation sigmoid --seed 42 --plot\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --layers <list>        Example: --layers 30 24 24 2\n";
    std::cout << "  --epochs <int>         Default: 50\n";
    std::cout << "  --batch_size <int>     Default: 8\n";
    std::cout << "  --lr <float>           Default: 0.1\n";
    std::cout << "  --activation <name>    sigmoid | tanh | relu (default: sigmoid)\n";
    std::cout << "  --seed <int>           Default: 42\n";
    std::cout << "  --scaler_out <path>    Default: models/scaler.txt\n";
    std::cout << "  --model_out <path>     Default: models/model.txt\n";
    std::cout << "  --history_out <path>   Default: models/history.csv\n";
    std::cout << "  --plot                 Generate loss/accuracy plots after training\n";
}

bool parse_args(int argc, char** argv, Options& out, std::string& error) {
    if (argc < 3) {
        error = "not enough arguments";
        return false;
    }

    out.train_path = argv[1];
    out.valid_path = argv[2];
    out.scaler_path = "models/scaler.txt";
    out.model_path = "models/model.txt";

    out.cfg.layers.clear();
    out.cfg.epochs = 50;
    out.cfg.batch_size = 8;
    out.cfg.learning_rate = 0.1;
    out.cfg.seed = 42;
    out.cfg.history_path = "models/history.csv";
    out.cfg.activation = mlp::Activation::Sigmoid;
    out.do_plot = false;

    int arg = 3;
    while (arg < argc) {
        const std::string key = argv[arg];
        if (key == "--layers") {
            out.cfg.layers.clear();
            ++arg;
            while (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
                out.cfg.layers.push_back(std::atoi(argv[arg]));
                ++arg;
            }
        } else if (key == "--epochs" && arg + 1 < argc) {
            out.cfg.epochs = std::atoi(argv[arg + 1]);
            arg += 2;
        } else if (key == "--batch_size" && arg + 1 < argc) {
            out.cfg.batch_size = std::atoi(argv[arg + 1]);
            arg += 2;
        } else if (key == "--lr" && arg + 1 < argc) {
            out.cfg.learning_rate = std::atof(argv[arg + 1]);
            arg += 2;
        } else if (key == "--activation" && arg + 1 < argc) {
            const std::string name = argv[arg + 1];
            if (name == "sigmoid") out.cfg.activation = mlp::Activation::Sigmoid;
            else if (name == "tanh") out.cfg.activation = mlp::Activation::Tanh;
            else if (name == "relu") out.cfg.activation = mlp::Activation::Relu;
            else {
                error = "Unknown activation: " + name;
                return false;
            }
            arg += 2;
        } else if (key == "--seed" && arg + 1 < argc) {
            out.cfg.seed = static_cast<unsigned int>(std::atoi(argv[arg + 1]));
            arg += 2;
        } else if (key == "--scaler_out" && arg + 1 < argc) {
            out.scaler_path = argv[arg + 1];
            arg += 2;
        } else if (key == "--model_out" && arg + 1 < argc) {
            out.model_path = argv[arg + 1];
            arg += 2;
        } else if (key == "--history_out" && arg + 1 < argc) {
            out.cfg.history_path = argv[arg + 1];
            arg += 2;
        } else if (key == "--plot") {
            out.do_plot = true;
            ++arg;
        } else {
            error = "Unknown or incomplete option: " + key;
            return false;
        }
    }

    return true;
}

} // namespace train_cli
