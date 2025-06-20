#include <vector>

#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "config.h"
#include "functions.h"

NeuralNetwork::NeuralNetwork(const bool load_weights) {
    if (load_weights) {
        weights = readWeights();
    } else {
        weights = getRandomMatrix();
    }
}

void NeuralNetwork::doOneBatch(const size_t current_batch_index, const std::vector<DataPoint> &dataset_train) {
    const size_t start_index = static_cast<size_t>(current_batch_index) * BATCH_SIZE;
    const size_t end_index = std::min(static_cast<size_t>(current_batch_index + 1) * BATCH_SIZE, dataset_train.size());

    const WEIGHT_SHAPE delta_matrix = getDeltaMatrix(start_index, end_index, dataset_train);
    adjustWeights(delta_matrix);
}

WEIGHT_SHAPE NeuralNetwork::getDeltaMatrix(size_t start_index, size_t end_index, const std::vector<DataPoint> &dataset_train) const {
    WEIGHT_SHAPE delta_matrix{};

    for (size_t current_datapoint_index = start_index; current_datapoint_index < end_index; ++current_datapoint_index) {
        const IMAGE_SHAPE &x = dataset_train[current_datapoint_index].pixels;
        const int real_label = dataset_train[current_datapoint_index].label;

        NN_OUTPUT_SHAPE processed_output = sigmoid(multiplyInputVectorWithWeights(x));

        for (size_t current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
            const int target_output = (real_label == static_cast<int>(current_digit)) ? 1 : 0;
            const double error_term = LEARNING_RATE * (target_output - processed_output[current_digit]);

            for (size_t current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
                delta_matrix[current_digit][current_weight_index] += error_term * x[current_weight_index];
            }
        }
    }
    return delta_matrix;
}

void NeuralNetwork::adjustWeights(const WEIGHT_SHAPE &delta_matrix) {
    for (size_t current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
        for (size_t current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
            weights[current_digit][current_weight_index] += delta_matrix[current_digit][current_weight_index];
        }
    }
}

NN_OUTPUT_SHAPE NeuralNetwork::multiplyInputVectorWithWeights(const IMAGE_SHAPE &input_data) const {

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

int NeuralNetwork::getPrediction(const IMAGE_SHAPE &input_data) const { return indexOfMax(multiplyInputVectorWithWeights(input_data)); }

WEIGHT_SHAPE NeuralNetwork::getWeights() const { return weights; }
