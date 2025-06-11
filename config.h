#ifndef CONFIG_H
#define CONFIG_H

constexpr double EXECUTIONS_QUANTITY = 2;

constexpr int IMAGE_SIZE = 28;
constexpr int NN_INPUT_SIZE = 784; // 28x28
constexpr int NN_OUTPUT_SIZE = 10; // 10 digits

using WEIGHT_SHAPE = std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE>;
using NN_OUTPUT_SHAPE = std::array<double, NN_OUTPUT_SIZE>;
using IMAGE_SHAPE = std::array<uint8_t, NN_INPUT_SIZE>;

constexpr int BATCH_SIZE = 64;
constexpr int EPOCHS_NUMBER = 10;
constexpr double LEARNING_RATE = 0.001;

#include <string>
const std::string WEIGHTS_PATH = "weights.csv";

const std::string DATASET_TRAIN_PATH = "dataset/mnist_train.csv";
const std::string DATASET_TEST_PATH = "dataset/mnist_test.csv";

#endif // CONFIG_H
