#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>

std::vector<std::vector<uint8_t>> readCSV(const std::string& filepath) {
    std::vector<std::vector<uint8_t>> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line); // ignore header

    while (std::getline(file, line)) {
        std::vector<uint8_t> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                unsigned int num = std::stoul(value);
                if (num > 255) {
                    std::cerr << "Value out of range: " << num << std::endl;
                    continue;
                }
                row.push_back(static_cast<uint8_t>(num));
            } catch (...) {
                std::cerr << "Invalid integer: " << value << std::endl;
            }
        }

        if (row.size() == 785) {
            data.push_back(row);
        } else {
            std::cerr << "Invalid row length: " << row.size() << " (expected 785)" << std::endl;
        }
    }

    file.close();
    return data;
}


int main() {

    std::cout << "===> Begin to load dataset" << std::endl;
    std::vector<std::vector<uint8_t>> dataset_test = readCSV("dataset/mnist_test.csv");
    std::vector<std::vector<uint8_t>> dataset_train = readCSV("dataset/mnist_train.csv");

    std::cout << "=> Finished. train set size : " << dataset_train.size() << " data points. " << " test set size : " << dataset_test.size() << " rows." << std::endl;

    return 0;
}
