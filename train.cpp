#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset" << std::endl;
    std::vector<DataPoint> dataset_train = readDataset(DATASET_TRAIN_PATH);
    std::cout << "Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    // show_dataset_element(dataset_train[12]);

    std::cout << "===================> Begin to train the model" << std::endl;
    const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> model = getTrainedModel(dataset_train);

    std::cout << "===================> Begin to save the weights" << std::endl;
    saveWeights(model, "weights.csv");

    return 0;
}
