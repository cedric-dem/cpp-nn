#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset train" << std::endl;
    std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_train = readDataset(DATASET_TRAIN_PATH);
    std::cout << "Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    std::cout << "===================> Begin to load dataset test" << std::endl;
    const std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_test = readDataset(DATASET_TEST_PATH);
    std::cout << "=> Finished loading test set, size " << dataset_test.size() << " . " << std::endl;

    for (int current_train= 0; current_train <= 10; ++current_train) {

        std::cout << "===================> Begin to train the model" << std::endl;
        const std::vector<std::vector<double>> model = get_trained_model(dataset_train);

        std::cout << "===================> Begin evaluation on the test set" << std::endl;
        evaluate_model(model, dataset_test);
    }

    return 0;
}
