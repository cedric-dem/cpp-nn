#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "config.h"
#include "functions.h"

bool parseLine(const std::string &line, DataPoint &outDataPoint) {
    std::stringstream ss(line);
    std::string value;
    std::vector<uint8_t> row;
    row.reserve(NN_INPUT_SIZE + 1);

    while (std::getline(ss, value, ',')) {
        try {
            unsigned int num = std::stoul(value);
            if (num > 255) {
                std::cerr << "Value out of range (0–255): " << num << std::endl;
                return false;
            }
            row.push_back(static_cast<uint8_t>(num));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Invalid number: " << value << std::endl;
            return false;
        } catch (const std::out_of_range &e) {
            std::cerr << "Number too large: " << value << std::endl;
            return false;
        }
    }

    if (row.size() != NN_INPUT_SIZE + 1) {
        std::cerr << "Incorrect number of values in line: expected " << NN_INPUT_SIZE + 1 << ", got " << row.size() << std::endl;
        return false;
    }

    outDataPoint.label = row[0];
    std::copy_n(row.begin() + 1, NN_INPUT_SIZE, outDataPoint.pixels.begin());
    return true;
}

std::vector<DataPoint> readDataset(const std::string &filepath) {
    std::vector<DataPoint> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        DataPoint dp;
        if (parseLine(line, dp)) {
            data.push_back(dp);
        } else {
            std::cerr << "Skipping malformed line." << std::endl;
        }
    }

    return data;
}

bool parseWeightLine(const std::string &line, size_t row, std::array<double, NN_INPUT_SIZE> &outRow) {
    std::stringstream ss(line);
    std::string cell;
    size_t col = 0;

    while (std::getline(ss, cell, ',') && col < NN_INPUT_SIZE) {
        try {
            outRow[col] = std::stod(cell);
        } catch (const std::invalid_argument &) {
            std::cerr << "Invalid value at [" << row << "][" << col << "]: '" << cell << "'\n";
            outRow[col] = 0.0;
        } catch (const std::out_of_range &) {
            std::cerr << "Out-of-range value at [" << row << "][" << col << "]: '" << cell << "'\n";
            outRow[col] = 0.0;
        }
        ++col;
    }

    return col == NN_INPUT_SIZE;
}

WEIGHT_SHAPE readWeights() {
    WEIGHT_SHAPE data{};

    std::ifstream file(WEIGHTS_PATH);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << WEIGHTS_PATH << std::endl;
        return data;
    }

    std::string line;
    size_t row = 0;

    while (std::getline(file, line) && row < NN_OUTPUT_SIZE) {
        if (!parseWeightLine(line, row, data[row])) {
            std::cerr << "Warning: Line " << row << " has incorrect number of columns.\n";
        }
        ++row;
    }

    return data;
}

NN_OUTPUT_SHAPE sigmoid(const NN_OUTPUT_SHAPE &inp) {
    NN_OUTPUT_SHAPE out{};
    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        double x = std::clamp(inp[i], -500.0, 500.0); // anti-overflow
        out[i] = 1.0 / (1.0 + std::exp(-x));
    }
    return out;
}

WEIGHT_SHAPE getRandomMatrix() {
    WEIGHT_SHAPE result{};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        for (size_t j = 0; j < NN_INPUT_SIZE; ++j) {
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

NeuralNetwork getTrainedModel(std::vector<DataPoint> &dataset_train) {

    const size_t number_of_batches = static_cast<size_t>(std::ceil(static_cast<double>(dataset_train.size()) / BATCH_SIZE));

    std::cout << "==> Number of batches : " << number_of_batches << std::endl;

    NeuralNetwork model(false);

    for (int epoch = 1; epoch <= EPOCHS_NUMBER; ++epoch) {
        shuffleDataset(dataset_train);
        std::cout << "=> Epoch " << epoch << "/" << EPOCHS_NUMBER << std::endl;

        for (size_t batch_idx = 0; batch_idx < number_of_batches; ++batch_idx) {
            model.doOneBatch(batch_idx, dataset_train);
        }
    }
    return model;
}

void saveWeights(const WEIGHT_SHAPE &model, const std::string &filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filepath << std::endl;
        return;
    }

    for (const auto &row : model) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << ",";
            }
        }
        file << "\n";
    }

    if (!file) {
        std::cerr << "Error: writing to file " << filepath << " failed." << std::endl;
        return;
    }

    std::cout << "==> Finished writing weights to " << filepath << std::endl;
}

int indexOfMax(const NN_OUTPUT_SHAPE &output) {
    size_t max_index = 0;
    double max_value = output[0];

    for (size_t i = 1; i < NN_OUTPUT_SIZE; ++i) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }

    return static_cast<int>(max_index);
}

void displayConfusionMatrix(const std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> &data, size_t dataset_size) {
    std::cout << "=================================== Confusion Matrix ===================================" << std::endl;

    // Display top banner with real answers
    std::cout << "********************************** Predicted by the model ****************************" << std::endl << "     ";
    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        std::cout << std::setw(7) << i;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        std::cout << std::setw(7) << ("Real " + std::to_string(i));
        for (size_t j = 0; j < NN_OUTPUT_SIZE; ++j) {
            if (data[i][j] > 0) {
                double percent = 100.0 * data[i][j] / static_cast<double>(dataset_size);
                std::cout << std::setw(7) << std::fixed << std::setprecision(2) << percent << "%";

            } else {
                std::cout << std::setw(7) << "   -----";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "========================================================================================" << std::endl;
}

double evaluateModel(const NeuralNetwork &model, const std::vector<DataPoint> &dataset, bool show_confusion_matrix) {
    size_t good_predictions = 0;

    std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> confusion_matrix{};

    for (const auto &dp : dataset) {
        const size_t current_real = dp.label;
        const size_t current_prediction = model.getPrediction(dp.pixels);

        if (current_real == current_prediction) {
            ++good_predictions;
        }

        if (current_real < NN_OUTPUT_SIZE && current_prediction < NN_OUTPUT_SIZE) {
            confusion_matrix[current_real][current_prediction] += 1;
        } else {
            std::cerr << "Warning: label or prediction out of range: label=" << current_real << ", prediction=" << current_prediction << std::endl;
        }
    }

    if (show_confusion_matrix) {
        displayConfusionMatrix(confusion_matrix, dataset.size());
    }

    double percentage = 100.0 * static_cast<double>(good_predictions) / dataset.size();
    std::cout << "==> Good predictions : " << good_predictions << "/" << dataset.size() << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;

    return percentage;
}