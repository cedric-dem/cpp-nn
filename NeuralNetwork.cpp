#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <vector>

#include "config.h"
#include "functions.h"

#include "NeuralNetwork.h"

#include "DataPoint.h"

NeuralNetwork::NeuralNetwork(const bool load_weights) {
    if (load_weights) {
        weights = readWeights();
    } else {
        weights = getRandomMatrix();
    }
}

void NeuralNetwork::doOneBatch(const int current_batch_index, const std::vector<DataPoint> &dataset_train) {

    const int start_index = current_batch_index * BATCH_SIZE;
    const int end_index = std::min(((current_batch_index + 1) * BATCH_SIZE), static_cast<int>(dataset_train.size()));

    const WEIGHT_SHAPE delta_matrix = getDeltaMatrix(start_index, end_index, dataset_train);

    adjustWeights(delta_matrix);
}

WEIGHT_SHAPE NeuralNetwork::getDeltaMatrix(const int start_index, const int end_index, const std::vector<DataPoint> &dataset_train) const {

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

void NeuralNetwork::adjustWeights(const WEIGHT_SHAPE &delta_matrix) {
    for (int current_digit = 0; current_digit < NN_OUTPUT_SIZE; ++current_digit) {
        for (int current_weight_index = 0; current_weight_index < NN_INPUT_SIZE; ++current_weight_index) {
            weights[current_digit][current_weight_index] += delta_matrix[current_digit][current_weight_index];
        }
    }
}

int NeuralNetwork::getPrediction(const IMAGE_SHAPE &input_data) const {
    // TODO activation function ?
    const NN_OUTPUT_SHAPE output = multiplyInputVectorWithWeights(input_data, weights);

    return indexOfMax(output);
}

WEIGHT_SHAPE NeuralNetwork::getWeights() const { return weights; }
