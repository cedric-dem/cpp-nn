#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "functions.h"

std::vector<DataPoint> readDataset(const std::string &filepath) {
    std::vector<DataPoint> data;
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

            DataPoint to_add;
            // TODO clean that code
            uint8_t label = row[0];

            to_add.label = label;

            std::array<uint8_t, NN_INPUT_SIZE> input_data_array{};
            for (int i = 0; i < NN_INPUT_SIZE; i++) {
                input_data_array[i] = row[i + 1];
            }

            to_add.pixels = input_data_array;
            data.push_back(to_add);
        } else {
            std::cerr << "Invalid row length: " << row.size() << " (expected 785)" << std::endl;
        }
    }

    file.close();
    return data;
}

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> readWeights() {
    std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> data{};

    std::ifstream file(WEIGHTS_PATH);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << WEIGHTS_PATH << std::endl;
        return data;
    }

    std::string line;
    size_t row = 0;

    while (std::getline(file, line) && row < NN_OUTPUT_SIZE) {
        std::stringstream ss(line);
        std::string cell;
        size_t col = 0;

        while (std::getline(ss, cell, ',') && col < NN_INPUT_SIZE) {
            try {
                data[row][col] = std::stod(cell);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error, location ; " << row << " ; " << col << ": '" << cell << "'\n";
                data[row][col] = 0.0;
            }
            ++col;
        }

        ++row;
    }

    file.close();
    return data;
}

void displayMatrix(const std::array<uint8_t, NN_INPUT_SIZE> &data, const uint8_t size_a, const uint8_t size_b) {

    for (size_t row = 0; row < size_a; ++row) {
        for (size_t col = 0; col < size_b; ++col) {
            std::cout << static_cast<int>(data[row * size_a + col]) << ' ';
        }
        std::cout << '\n';
    }
}

void showDatasetElement(const DataPoint &dataset_elem) {
    std::cout << "======> Displaying sample digit " << static_cast<int>(dataset_elem.label) << std::endl;

    displayMatrix(dataset_elem.pixels, IMAGE_SIZE, IMAGE_SIZE);
}

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getRandomMatrix() {
    std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> result{};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {    // TO verify
        for (int j = 0; j < NN_INPUT_SIZE; ++j) { // TO verify
            result[i][j] = d(gen);
        }
    }

    return result;
}

void shuffleDataset(std::vector<DataPoint> &dataset) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(dataset.begin(), dataset.end(), gen);
}

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getTrainedModel(std::vector<DataPoint> &dataset_train) {

    std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> current_weights = getRandomMatrix();

    const int number_of_batches = static_cast<int>(std::ceil(static_cast<double>(dataset_train.size()) / BATCH_SIZE));

    std::cout << "==> N batches : " << number_of_batches << std::endl;

    for (int current_epoch = 1; current_epoch <= EPOCHS_NUMBER; ++current_epoch) {
        shuffleDataset(dataset_train);

        std::cout << "=====> current Epoch : " << current_epoch << "/" << EPOCHS_NUMBER << "<=============" << std::endl;

        for (int current_batch_index = 0; current_batch_index < number_of_batches; ++current_batch_index) {
            // std::cout << "=====> current batch : " << current_batch_index << "/" << number_of_batches << std::endl;
            batch(current_batch_index, dataset_train, current_weights);
        }
    }
    return current_weights;
}

void batch(const int current_batch_index, const std::vector<DataPoint> &dataset_train, std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights) {

    const int start_index = current_batch_index * BATCH_SIZE;
    const int end_index = std::min(((current_batch_index + 1) * BATCH_SIZE), static_cast<int>(dataset_train.size()));

    std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> delta_matrix = getDeltaMatrix(start_index, end_index, dataset_train, current_weights);

    adjustWeights(current_weights, delta_matrix);
}

void adjustWeights(std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &delta_matrix) {
    for (int current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
        for (int current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
            current_weights[current_digit][current_weight_index] += delta_matrix[current_digit][current_weight_index];
        }
    }
}

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getDeltaMatrix(const int start_index, const int end_index, const std::vector<DataPoint> &dataset_train, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights) {

    // double delta_matrix[NN_OUTPUT_SIZE][NN_INPUT_SIZE] = {0};
    std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> delta_matrix{};

    int current_real_output;
    for (int current_datapoint_index = start_index; current_datapoint_index < end_index; ++current_datapoint_index) {

        // input data
        std::array<uint8_t, NN_INPUT_SIZE> x = dataset_train[current_datapoint_index].pixels;

        // real answer
        const int real_label = dataset_train[current_datapoint_index].label;

        // prediction
        std::array<double, NN_OUTPUT_SIZE> raw_output = multiplyInputVectorWithWeights(x, current_weights);

        // std::array<double, NN_OUTPUT_SIZE>  processed_output = biggest_1_else_0(raw_output);
        //  std::array<double, NN_OUTPUT_SIZE>  processed_output = sigmoid(raw_output);
        std::array<double, NN_OUTPUT_SIZE> processed_output = fBinary(raw_output);

        // adjust delta weight
        for (int current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
            if (real_label == current_digit) {
                current_real_output = 1;
            } else {
                current_real_output = 0;
            }

            for (int current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
                // if not good (or always ?)
                delta_matrix[current_digit][current_weight_index] += LEARNING_RATE * (current_real_output - processed_output[current_digit]) * x[current_weight_index];
            }
        }
    }
    return delta_matrix;
}

std::array<double, NN_OUTPUT_SIZE> biggest1Else0(const std::array<double, NN_OUTPUT_SIZE> &inp) {
    std::array<double, NN_OUTPUT_SIZE> out{};
    out[indexOfMax(inp)] = 1;
    return out;
}

std::array<double, NN_OUTPUT_SIZE> sigmoid(const std::array<double, NN_OUTPUT_SIZE> &inp) {
    std::array<double, NN_OUTPUT_SIZE> out{};

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        out[i] = 1.0 / (1.0 + std::exp(-inp[i]));
    }

    return out;
}

std::array<double, NN_OUTPUT_SIZE> fBinary(const std::array<double, NN_OUTPUT_SIZE> &inp) {
    std::array<double, NN_OUTPUT_SIZE> out{};

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        if (inp[i] >= 0) {
            out[i] = 1;
        } else {
            out[i] = 0;
        }
    }

    return out;
}

void saveWeights(const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &model, const std::string &filepath) {
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

std::array<double, NN_OUTPUT_SIZE> multiplyInputVectorWithWeights(const std::array<uint8_t, NN_INPUT_SIZE> &input_data, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights) {

    const size_t num_rows = NN_OUTPUT_SIZE; // TO Verify
    const size_t num_cols = NN_INPUT_SIZE;  // TO verify

    if (input_data.size() != num_cols) {
        throw std::invalid_argument("Matrix/vector size are incompatible");
    }

    std::array<double, NN_OUTPUT_SIZE> result{};

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

int indexOfMax(const std::array<double, NN_OUTPUT_SIZE> &output) {
    int max_index = 0;
    double max_value = output[0];

    for (int i = 1; i < NN_OUTPUT_SIZE; ++i) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }

    return max_index;
}

int getPrediction(const std::array<uint8_t, NN_INPUT_SIZE> &input_data, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights) {
    // TODO activation function ?
    const std::array<double, NN_OUTPUT_SIZE> output = multiplyInputVectorWithWeights(input_data, weights);

    return indexOfMax(output);
}

double evaluateModel(const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights, const std::vector<DataPoint> &dataset) {
    int good_predictions = 0;

    int current_prediction;
    int current_real;

    // for elem in dataset
    for (size_t i = 0; i < dataset.size(); ++i) {
        current_real = dataset[i].label;

        current_prediction = getPrediction(dataset[i].pixels, weights);

        if (current_real == current_prediction) {
            good_predictions += 1;
        }
    }

    const double percentage = 100.0 * static_cast<double>(good_predictions) / dataset.size();
    std::cout << "=> good predictions : " << good_predictions << "/" << dataset.size() << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    return percentage;
}