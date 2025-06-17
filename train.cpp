#include <iostream>
#include <vector>

#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "config.h"
#include "functions.h"

int main() {

    std::cout << "===================> Begin to load dataset" << std::endl;
    std::vector<DataPoint> dataset_train = readDataset(DATASET_TRAIN_PATH);
    std::cout << "Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    // show_dataset_element(dataset_train[12]);

    std::cout << "===================> Begin to train the model" << std::endl;
    const NeuralNetwork model = getTrainedModel(dataset_train);

    std::cout << "===================> Begin to save the weights" << std::endl;
    saveWeights(model.getWeights(), "weights.csv");

    return 0;
}
