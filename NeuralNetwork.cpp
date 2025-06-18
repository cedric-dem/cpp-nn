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

void NeuralNetwork::doOneBatch(const int current_batch_index, const std::vector<DataPoint> &dataset_train) {
    const size_t start_index = static_cast<size_t>(current_batch_index) * BATCH_SIZE;
    const size_t end_index = std::min(static_cast<size_t>(current_batch_index + 1) * BATCH_SIZE, dataset_train.size());

    const WEIGHT_SHAPE delta_matrix = getDeltaMatrix(start_index, end_index, dataset_train);
    adjustWeights(delta_matrix);
}

WEIGHT_SHAPE NeuralNetwork::getDeltaMatrix(const int start_index, const int end_index, const std::vector<DataPoint> &dataset_train) const {
    WEIGHT_SHAPE delta_matrix{};

    int current_real_output;
    for (int current_datapoint_index = start_index; current_datapoint_index < end_index; ++current_datapoint_index) {

        // input data
        IMAGE_SHAPE x = dataset_train[current_datapoint_index].pixels;

        // real answer
        const int real_label = dataset_train[current_datapoint_index].label;

        // prediction
        NN_OUTPUT_SHAPE processed_output = sigmoid(multiplyInputVectorWithWeights(x));

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

void NeuralNetwork::adjustWeights(const WEIGHT_SHAPE &delta_matrix) {
    for (int current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
        for (int current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
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
