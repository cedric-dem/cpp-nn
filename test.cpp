#include <iostream>
#include <vector>

#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "config.h"
#include "functions.h"

int main() {

    std::cout << "====> Begin to load dataset" << std::endl;
    const std::vector<DataPoint> dataset_test = readDataset(DATASET_TEST_PATH);
    std::cout << "====> Finished loading test set, size " << dataset_test.size() << " . " << std::endl;

    std::cout << "====> Begin to load weights" << std::endl;
    const NeuralNetwork model_to_test = NeuralNetwork(true);

    std::cout << "====> Begin evaluation on the test set" << std::endl;
    evaluateModel(model_to_test, dataset_test);

    return 0;
}
