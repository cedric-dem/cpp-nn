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

class NeuralNetwork {
  private:
    WEIGHT_SHAPE weights{};

  public:
    NeuralNetwork() {
        // Initialize weights at random
        weights = getRandomMatrix();
    }

    void batch(const int current_batch_index, const std::vector<DataPoint> &dataset_train) {

        const int start_index = current_batch_index * BATCH_SIZE;
        const int end_index = std::min(((current_batch_index + 1) * BATCH_SIZE), static_cast<int>(dataset_train.size()));

        const WEIGHT_SHAPE delta_matrix = getDeltaMatrix(start_index, end_index, dataset_train);

        adjustWeights(delta_matrix);
    }

    WEIGHT_SHAPE getDeltaMatrix(const int start_index, const int end_index, const std::vector<DataPoint> &dataset_train) const {

        // double delta_matrix[NN_OUTPUT_SIZE][NN_INPUT_SIZE] = {0};
        WEIGHT_SHAPE delta_matrix{};

        int current_real_output;
        for (int current_datapoint_index = start_index; current_datapoint_index < end_index; ++current_datapoint_index) {

            // input data
            IMAGE_SHAPE x = dataset_train[current_datapoint_index].pixels;

            // real answer
            const int real_label = dataset_train[current_datapoint_index].label;

            // prediction
            NN_OUTPUT_SHAPE raw_output = multiplyInputVectorWithWeights(x, weights);

            // NN_OUTPUT_SHAPE  processed_output = biggest_1_else_0(raw_output);
            //  NN_OUTPUT_SHAPE  processed_output = sigmoid(raw_output);
            NN_OUTPUT_SHAPE processed_output = fBinary(raw_output);

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

    void adjustWeights(const WEIGHT_SHAPE &delta_matrix) {
        for (int current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
            for (int current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
                weights[current_digit][current_weight_index] += delta_matrix[current_digit][current_weight_index];
            }
        }
    }

    WEIGHT_SHAPE getWeights() const { return weights; }
};

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
        std::stringstream ss(line);
        std::string value;
        std::vector<uint8_t> row;
        row.reserve(NN_INPUT_SIZE + 1); // label + pixels

        while (std::getline(ss, value, ',')) {
            try {
                unsigned int num = std::stoul(value);
                if (num > 255) {
                    std::cerr << "Value not in interval" << num << std::endl;
                    row.clear();
                    break;
                }
                row.push_back(static_cast<uint8_t>(num));
            } catch (...) {
                std::cerr << "Invalid value " << value << std::endl;
                row.clear();
                break;
            }
        }

        if (row.size() != NN_INPUT_SIZE + 1) {
            std::cerr << "Invalid line length " << row.size() << std::endl;
            continue;
        }

        DataPoint dp;
        dp.label = row[0];
        std::copy_n(row.begin() + 1, NN_INPUT_SIZE, dp.pixels.begin());

        data.push_back(dp);
    }

    return data;
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

void displayMatrix(const IMAGE_SHAPE &data, const uint8_t size_a, const uint8_t size_b) {

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

WEIGHT_SHAPE getRandomMatrix() {
    WEIGHT_SHAPE result{};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        for (int j = 0; j < NN_INPUT_SIZE; ++j) {
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

WEIGHT_SHAPE getTrainedModel(std::vector<DataPoint> &dataset_train) {

    const int number_of_batches = static_cast<int>(std::ceil(static_cast<double>(dataset_train.size()) / BATCH_SIZE));

    std::cout << "==> N batches : " << number_of_batches << std::endl;

    NeuralNetwork this_model = NeuralNetwork();

    for (int current_epoch = 1; current_epoch <= EPOCHS_NUMBER; ++current_epoch) {
        shuffleDataset(dataset_train);

        std::cout << "=====> current Epoch : " << current_epoch << "/" << EPOCHS_NUMBER << "<=============" << std::endl;

        for (int current_batch_index = 0; current_batch_index < number_of_batches; ++current_batch_index) {
            // std::cout << "=====> current batch : " << current_batch_index << "/" << number_of_batches << std::endl;
            this_model.batch(current_batch_index, dataset_train);
        }
    }
    return this_model.getWeights();
}

NN_OUTPUT_SHAPE biggest1Else0(const NN_OUTPUT_SHAPE &inp) {
    NN_OUTPUT_SHAPE out{};
    out[indexOfMax(inp)] = 1;
    return out;
}

NN_OUTPUT_SHAPE sigmoid(const NN_OUTPUT_SHAPE &inp) {
    NN_OUTPUT_SHAPE out{};

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        out[i] = 1.0 / (1.0 + std::exp(-inp[i]));
    }

    return out;
}

NN_OUTPUT_SHAPE fBinary(const NN_OUTPUT_SHAPE &inp) {
    NN_OUTPUT_SHAPE out{};

    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        if (inp[i] >= 0) {
            out[i] = 1;
        } else {
            out[i] = 0;
        }
    }

    return out;
}

void saveWeights(const WEIGHT_SHAPE &model, const std::string &filepath) {
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

NN_OUTPUT_SHAPE multiplyInputVectorWithWeights(const IMAGE_SHAPE &input_data, const WEIGHT_SHAPE &weights) {

    NN_OUTPUT_SHAPE result{};

    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < NN_INPUT_SIZE; ++j) {
            sum += static_cast<double>(input_data[j]) * weights[i][j];
        }
        result[i] = sum;
    }

    return result;
}

int indexOfMax(const NN_OUTPUT_SHAPE &output) {
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

int getPrediction(const IMAGE_SHAPE &input_data, const WEIGHT_SHAPE &weights) {
    // TODO activation function ?
    const NN_OUTPUT_SHAPE output = multiplyInputVectorWithWeights(input_data, weights);

    return indexOfMax(output);
}

void displayConfusionMatrix(const std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> &data, const int dataset_size) {
    std::cout << "=================================== Confusion Matrix ===================================" << std::endl;

    // Display top banner with real answers
    std::cout << "********************************** Predicted by the model ****************************" << std::endl << "     ";
    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        std::cout << i << "       ";
    }
    std::cout << std::endl;

    // Matrix itself
    for (size_t i = 0; i < NN_OUTPUT_SIZE; ++i) {
        std::cout << "   Real   "[i] << " " << "0123456789"[i] << "  ";
        for (size_t j = 0; j < NN_OUTPUT_SIZE; ++j) {
            if (data[i][j] > 0) {
                std::cout << std::fixed << std::setprecision(2) << 100.0 * static_cast<double>(data[i][j]) / dataset_size << "%   ";
            } else {
                std::cout << "        ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "========================================================================================" << std::endl;
}

double evaluateModel(const WEIGHT_SHAPE &weights, const std::vector<DataPoint> &dataset, bool show_confusion_matrix) {
    int good_predictions = 0;

    int current_prediction;
    int current_real;

    std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> confusion_matrix{};

    // for elem in dataset
    for (size_t i = 0; i < dataset.size(); ++i) {
        current_real = dataset[i].label;

        current_prediction = getPrediction(dataset[i].pixels, weights);

        if (current_real == current_prediction) {
            good_predictions += 1;
        }

        confusion_matrix[current_real][current_prediction] += 1;
    }

    if (show_confusion_matrix) {
        displayConfusionMatrix(confusion_matrix, static_cast<int>(dataset.size()));
    }

    const double percentage = 100.0 * static_cast<double>(good_predictions) / dataset.size();
    std::cout << "=> good predictions : " << good_predictions << "/" << dataset.size() << " (" << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    return percentage;
}