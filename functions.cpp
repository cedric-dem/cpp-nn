#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
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
            std::vector<uint8_t> first784(row.begin(), row.begin() + 784);
            uint8_t last = row[784];
            data.emplace_back(std::move(first784), last);
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
    std::cout << "======> Displaying sample digit" << dataset_elem.second << std::endl;

    display_matrix(dataset_elem.first, 28, 28);
}

