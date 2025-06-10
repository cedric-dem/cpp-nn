#ifndef CONFIG_H
#define CONFIG_H

constexpr int IMAGE_SIZE = 28;
constexpr int NN_INPUT_SIZE = 784; // 28x28
constexpr int NN_OUTPUT_SIZE = 10; // 10 digits
constexpr int BATCH_SIZE = 32;     // Unused for now
constexpr int EPOCHS_NUMBER = 10;
constexpr double LEARNING_RATE = 0.001;

#include <string_view>
constexpr std::string_view WEIGHTS_PATH = "weights.csv";

constexpr std::string_view DATASET_TEST_PATH = "dataset/mnist_test.csv";
constexpr std::string_view DATASET_TRAIN_PATH = "dataset/mnist_train.csv";

#endif // CONFIG_H
