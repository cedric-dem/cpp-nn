#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset" << std::endl;
    std::vector<std::pair<std::vector<uint8_t>, uint8_t>>  dataset_train = readDataset("dataset/mnist_train.csv");
    std::cout << "Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    //show_dataset_element(dataset_train[12]);

    std::cout << "===================> Begin to train the model" << std::endl;
    std::vector<std::vector<double>> model = get_trained_model(dataset_train, 10);
    

    std::cout << "===================> Begin to save the weights" << std::endl;
    save_weights(model, "weights.csv");


    return 0;
}
