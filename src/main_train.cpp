#include "utils.hpp"
#include "mlp.hpp"
#include "dataset.hpp"
#include "mlp_io.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <train_csv> <valid_csv> [options] \n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " data_training.csv data_validation.csv \\\n";
    std::cout << "    --layers 30 24 24 2 --epochs 50 --batch_size 8 --lr 0.01 --activation sigmoid --plot\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --layers <list>       Example: --layers 30 24 24 2\n";
    std::cout << "  --epochs <int>        Default: 50\n";
    std::cout << "  --batch_size <int>    Default: 8\n";
    std::cout << "  --lr <float>          Default: 0.01\n";
    std::cout << "  --activation <name>   sigmoid | tanh | relu (default: sigmoid)\n";
    std::cout << "  --plot                Generate loss/accuracy plots after training\n";
}

static void ensure_models_dir() {
    std::error_code ec;
    std::filesystem::create_directories("models", ec);
    if (ec) {
        std::cout << "Warning: could not create models/ directory.\n";
    }
}

int main(int argc, char** argv) {
    utils::print_message("train");

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    
    std::string train_path = argv[1];
    std::string valid_path = argv[2];
    std::string scaler_path = "models/scaler.txt";
    std::string model_path = "models/model.txt";    

    // MLP CONFIGURATION
    mlp::MLPConfig cfg;
    cfg.layers.clear();
    cfg.epochs = 50;
    cfg.batch_size = 8;
    cfg.learning_rate = 0.1;
    cfg.history_path = "models/history.csv";
    cfg.activation = mlp::Activation::Sigmoid;
    bool do_plot = false;

    int arg = 3;
    while (arg < argc) {
        std::string key = argv[arg];
        if (key == "--layers") {
            cfg.layers.clear();
            arg++;
            while (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
                cfg.layers.push_back(std::atoi(argv[arg]));
                arg++;
            }
        } else if (key == "--epochs" && arg + 1 < argc) {
            cfg.epochs = std::atoi(argv[arg + 1]);
            arg += 2;
        } else if (key == "--batch_size" && arg + 1 < argc) {
            cfg.batch_size = std::atoi(argv[arg + 1]);
            arg += 2;
        } else if (key == "--lr" && arg + 1 < argc) {
            cfg.learning_rate = std::atof(argv[arg + 1]);
            arg += 2;
        } else if (key == "--activation" && arg + 1 < argc) {
            std::string name = argv[arg + 1];
            if (name == "sigmoid") cfg.activation = mlp::Activation::Sigmoid;
            else if (name == "tanh") cfg.activation = mlp::Activation::Tanh;
            else if (name == "relu") cfg.activation = mlp::Activation::Relu;
            else {
                std::cout << "Unknown activation: " << name << "\n";
                print_usage(argv[0]);
                return 1;
            }
            arg += 2;
        } else if (key == "--plot") {
            do_plot = true;
            arg++;
        } else {
            std::cout << "Unknown or incomplete option: " << key << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // if user don't give a MLP architecture, we use default architecture
    if (cfg.layers.empty()) {
        cfg.layers.push_back(30);
        cfg.layers.push_back(24);
        cfg.layers.push_back(24);
        cfg.layers.push_back(2);
    }

    // checking architecture  and parameters of MLP
    if (cfg.layers.size() < 4) {
        std::cout << "Error: need input + at least 2 hidden + output layers.\n";
        return 1;
    }
    if (cfg.layers.front() != 30) {
        std::cout << "Error: input layer must be 30 (number of features).\n";
        return 1;
    }
    if (cfg.layers.back() != 2) {
        std::cout << "Error: output layer must be 2 (softmax classes).\n";
        return 1;
    }
    if (cfg.epochs <= 0 || cfg.batch_size <= 0 || cfg.learning_rate <= 0.0) {
        std::cout << "Error: epochs, batch_size and lr must be positive.\n";
        return 1;
    }


    // DATA PREPROCESSING
    dataset::Dataset train = dataset::load_dataset(train_path);
    dataset::Dataset valid = dataset::load_dataset(valid_path);

    if (train.x.empty() || valid.x.empty()) {
        std::cout << "Error: failed to load datasets.\n";
        return 1;
    }

    dataset::Scaler s = dataset::fit_minmax(train);
    dataset::apply_minmax(train, s);
    dataset::apply_minmax(valid, s);

    // MODEL'S PATH CREATING
    ensure_models_dir();

    if (!dataset::save_scaler(scaler_path, s)) {
        std::cout << "Warning: could not save scaler to " << scaler_path << "\n";
    }


    // MODEL TRAINING
    mlp::MLP model;


    model.train(cfg, train, valid);
    if (!mlp::save_model(model, model_path)) {
        std::cout << "Warning: could not save model to " << model_path << "\n";
    } else {
        std::cout << "Model saved to: " << model_path << "\n";
    }

    // ACCURACY AND LOSS GRAPHS PLOTTING
    if (do_plot) {
        std::string cmd = "python3 scripts/plot.py " + cfg.history_path;
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::cout << "Warning: failed to generate plots (exit code " << rc << ")\n";
        }
    }
    return 0;
}
