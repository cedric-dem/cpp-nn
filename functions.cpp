#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include "functions.h"


std::vector<std::pair<std::vector<uint8_t>, uint8_t>> readDataset(const std::string& filepath) {
    std::vector<std::pair<std::vector<uint8_t>, uint8_t>> data;
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
            // Split into two parts
            uint8_t label = row[0];
            std::vector<uint8_t> input_data(row.begin()+1, row.begin() + 785);
            data.emplace_back(std::move(input_data), label);
        } else {
            std::cerr << "Invalid row length: " << row.size() << " (expected 785)" << std::endl;
        }
    }

    file.close();
    return data;
}

std::vector<std::vector<double>> readWeights(const std::string& filepath) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return data;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                double num = std::stod(value);
                row.push_back(num);
            } catch (...) {
                std::cerr << "Invalid double: " << value << std::endl;
            }
        }

        if (row.size() == 784) {        
            data.emplace_back(std::move(row));
        } else {
            std::cerr << "Invalid row length: " << row.size() << " (expected 784)" << std::endl;
        }
    }

    return data;
}


void display_matrix(const std::vector<uint8_t>& data, const uint8_t size_a, const uint8_t size_b) {
    if (data.size() != size_a*size_b) {
        std::cerr << "Error: vector size is not good" << std::endl;
        return;
    }
    
    for (size_t row = 0; row < size_a; ++row) {
        for (size_t col = 0; col < size_b; ++col) {
            std::cout << static_cast<int>(data[row * size_a + col]) << ' ';
        }
        std::cout << '\n';
    }
}


void show_dataset_element(const std::pair<std::vector<uint8_t>, uint8_t> dataset_elem){
    std::cout << "======> Displaying sample digit " << static_cast<int>(dataset_elem.second) << std::endl;


    display_matrix(dataset_elem.first, 28, 28);
}

std::vector<std::vector<double>> get_random_matrix(const int a, const int b) {
    std::vector<std::vector<double>> mat(a, std::vector<double>(b));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < b; ++j) {
            mat[i][j] = d(gen);
        }
    }

    return mat;
}

std::vector<std::vector<double>> get_trained_model(std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_train, const int epochs){
    std::vector<std::vector<double>> initial_weights = get_random_matrix(10, 784);

    for (int i = 1; i <= epochs; ++i) {
        std::cout << "===> current Epoch : " << i << "/" << epochs << std::endl;
        //TODO
    }

    return initial_weights;
}



void save_weights(std::vector<std::vector<double>>  model, const std::string& filepath){
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: " << filepath << std::endl;
        return;
    }

    for (const auto& row : model) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Finished writing weights" << std::endl;
}

