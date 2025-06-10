#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "functions.h"

std::vector<std::pair<std::vector<uint8_t>, uint8_t>> readDataset(const std::string &filepath) {
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
            std::vector input_data(row.begin() + 1, row.begin() + 785);
            data.emplace_back(std::move(input_data), label);
        } else {
            std::cerr << "Invalid row length: " << row.size() << " (expected 785)" << std::endl;
        }
    }

    file.close();
    return data;
}

std::vector<std::vector<double>> readWeights(const std::string &filepath) {
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

void display_matrix(const std::vector<uint8_t> &data, const uint8_t size_a, const uint8_t size_b) {
    if (data.size() != size_a * size_b) {
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

void show_dataset_element(const std::pair<std::vector<uint8_t>, uint8_t> &dataset_elem) {
    std::cout << "======> Displaying sample digit " << static_cast<int>(dataset_elem.second) << std::endl;

    display_matrix(dataset_elem.first, 28, 28);
}

std::vector<std::vector<double>> get_random_matrix(const int a, const int b) {
    std::vector mat(a, std::vector<double>(b));

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

std::vector<std::vector<double>> get_trained_model(const std::vector<std::pair<std::vector<uint8_t>, uint8_t>> &dataset_train, const int epochs) {
    std::vector<std::vector<double>> current_weights = get_random_matrix(10, 784);

    const double learning_rate = 0.01;
    int y;
    int y_hat;

    std::vector<uint8_t> y_vector(10, 0);
    std::vector<uint8_t> y_hat_vector(10, 0);

    std::vector<uint8_t> x;

    for (int current_epoch = 1; current_epoch <= epochs; ++current_epoch) {
        std::cout << "===> current Epoch : " << current_epoch << "/" << epochs << std::endl;

        // TODO batch instead
        for (int current_datapoint_index = 0; current_datapoint_index < dataset_train.size(); ++current_datapoint_index) {
            // real answer
            x = dataset_train[current_datapoint_index].first;

            // input data
            y = dataset_train[current_datapoint_index].second;
            std::fill(y_vector.begin(), y_vector.end(), 0);
            y_vector[y] = 1;

            // prediction
            y_hat = get_prediction(x, current_weights);
            std::fill(y_hat_vector.begin(), y_hat_vector.end(), 0);
            y_hat_vector[y_hat] = 1;

            // if not good (or always ?)

            // adjust weight
            for (int current_digit = 0; current_digit < 10; ++current_digit) {
                for (int current_weight_index = 0; current_weight_index < 784; ++current_weight_index) {
                    current_weights[current_digit][current_weight_index] += learning_rate * (y_vector[current_digit] - y_hat_vector[current_digit]) * x[current_weight_index];
                }
            }
        }
    }
    return current_weights;
}

void save_weights(const std::vector<std::vector<double>> &model, const std::string &filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: " << filepath << std::endl;
        return;
    }

    for (const auto &row : model) {
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

std::vector<double> multiply_input_vector_with_weights(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights) {

    const size_t num_rows = weights.size();
    const size_t num_cols = weights[0].size();

    if (input_data.size() != num_cols) {
        throw std::invalid_argument("Matrix/vector size are incompatible");
    }

    std::vector result(num_rows, 0.0);

    for (size_t i = 0; i < num_rows; ++i) {
        if (weights[i].size() != num_cols) {
            throw std::invalid_argument("Matrix is not a rectangle");
        }

        double sum = 0.0;
        for (size_t j = 0; j < num_cols; ++j) {
            sum += static_cast<double>(input_data[j]) * weights[i][j];
        }
        result[i] = sum;
    }

    return result;
}

int index_of_max(const std::vector<double> &output) {
    if (output.empty())
        return -1;

    int maxIndex = 0;
    double maxValue = output[0];

    for (int i = 1; i < static_cast<int>(output.size()); ++i) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

int get_prediction(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights) {
    // TODO activation function ?
    const std::vector<double> output = multiply_input_vector_with_weights(input_data, weights);

    const int index_max = index_of_max(output);

    return index_max;
}

void evaluate_model(const std::vector<std::vector<double>> &weights, const std::vector<std::pair<std::vector<uint8_t>, uint8_t>> &dataset) {
    int good_predictions = 0;

    int current_prediction;
    int current_real;

    // for elem in dataset
    for (size_t i = 0; i < dataset.size(); ++i) {
        current_real = dataset[i].second;
        current_prediction = get_prediction(dataset[i].first, weights);

        if (current_real == current_prediction) {
            good_predictions += 1;
        }
    }

    const double percentage = 100.0 * static_cast<double>(good_predictions) / dataset.size();
    std::cout << "=> good predictions : " << good_predictions << "/" << dataset.size() << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
}