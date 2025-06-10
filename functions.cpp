#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include "functions.h"


std::vector<std::pair<std::vector<uint8_t>, uint8_t>> readCSV(const std::string& filepath) {
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


std::vector<double> get_random_vector(const int size) {
    std::vector<double> vec(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        vec[i] = d(gen);
    }

    return vec;

}