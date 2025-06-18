#ifndef CONFIG_H
#define CONFIG_H

#include <array>
#include <string>

// === Neural network architecture parameters ===
constexpr int NN_INPUT_SIZE = 784; // 28x28 pixels input size
constexpr int NN_OUTPUT_SIZE = 10; // Number of output classes (digits 0-9)

// === Type aliases ===
using WEIGHT_SHAPE = std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE>;
using NN_OUTPUT_SHAPE = std::array<double, NN_OUTPUT_SIZE>;
using IMAGE_SHAPE = std::array<double, NN_INPUT_SIZE>;

// === Training parameters ===
constexpr int BATCH_SIZE = 64;
constexpr int EPOCHS_NUMBER = 2;
constexpr double LEARNING_RATE = 0.001;

// === File paths ===
namespace config {
constexpr const char *WEIGHTS_PATH = "weights.csv";
constexpr const char *DATASET_TRAIN_PATH = "dataset/mnist_train.csv";
constexpr const char *DATASET_TEST_PATH = "dataset/mnist_test.csv";
} // namespace config

#endif // CONFIG_H
