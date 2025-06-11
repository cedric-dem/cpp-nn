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
    const WEIGHT_SHAPE weights = getTrainedModel(dataset_train);

    std::cout << "===================> Begin to save the weights" << std::endl;
    saveWeights(weights, "weights.csv");

    return 0;
}
