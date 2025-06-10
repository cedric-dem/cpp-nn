#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>

#include "functions.h"


int main() {

    std::cout << "===> Begin to load dataset" << std::endl;

    std::vector<std::pair<std::vector<uint8_t>, uint8_t>>  dataset_train = readCSV("dataset/mnist_train.csv");
    std::cout << "=> Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    show_dataset_element(dataset_train[12]);

    return 0;
}
