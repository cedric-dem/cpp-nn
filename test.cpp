#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "config.h"
#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset" << std::endl;
    const std::vector<DataPoint> dataset_test = readDataset(DATASET_TEST_PATH);
    std::cout << "=> Finished loading test set, size " << dataset_test.size() << " . " << std::endl;

    // show_dataset_element(dataset_test[19]);

    std::cout << "===================> Begin to load weights" << std::endl;
    const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> weights = readWeights();
    std::cout << "=> Finished loading weights, size " << weights.size() << " . " << std::endl;

    std::cout << "===================> Begin evaluation on the test set" << std::endl;
    evaluateModel(weights, dataset_test);

    return 0;
}
