#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>

#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset" << std::endl;
    std::vector<std::pair<std::vector<uint8_t>, uint8_t>>  dataset_test = readDataset("dataset/mnist_test.csv");
    std::cout << "=> Finished loading test set, size " << dataset_test.size() << " . " << std::endl;

    // show_dataset_element(dataset_test[19]);

    std::cout << "===================> Begin to load weights" << std::endl;
    std::vector<std::vector<double>>   weights = readWeights("weights.csv");
    std::cout << "=> Finished loading weights, size " << weights.size() << " . " << std::endl;


    return 0;
}
