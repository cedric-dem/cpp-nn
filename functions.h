#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "config.h"

struct DataPoint {
    IMAGE_SHAPE pixels;
    uint8_t label;
};

std::vector<DataPoint> readDataset(const std::string &filepath);

WEIGHT_SHAPE readWeights();

void displayConfusionMatrix(const std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> &data);

void displayMatrix(const IMAGE_SHAPE &data, uint8_t size_a, uint8_t size_b);

void showDatasetElement(DataPoint dataset_elem);

WEIGHT_SHAPE getRandomMatrix();

WEIGHT_SHAPE getTrainedModel(std::vector<DataPoint> &dataset_train);

void saveWeights(const WEIGHT_SHAPE &model, const std::string &filepath);

double evaluateModel(const WEIGHT_SHAPE &weights, const std::vector<DataPoint> &dataset, bool show_confusion_matrix);

int getPrediction(const IMAGE_SHAPE &input_data, const WEIGHT_SHAPE &weights);

NN_OUTPUT_SHAPE multiplyInputVectorWithWeights(const IMAGE_SHAPE &input_data, const WEIGHT_SHAPE &weights);

int indexOfMax(const NN_OUTPUT_SHAPE &output);

void shuffleDataset(std::vector<DataPoint> &dataset);

void batch(int current_batch_index, const std::vector<DataPoint> &dataset_train, WEIGHT_SHAPE &current_weights);

WEIGHT_SHAPE getDeltaMatrix(int start_index, int end_index, const std::vector<DataPoint> &dataset_train, const WEIGHT_SHAPE &current_weights);
void adjustWeights(WEIGHT_SHAPE &current_weights, const WEIGHT_SHAPE &delta_matrix);

NN_OUTPUT_SHAPE biggest1Else0(const NN_OUTPUT_SHAPE &inp);
NN_OUTPUT_SHAPE sigmoid(const NN_OUTPUT_SHAPE &inp);
NN_OUTPUT_SHAPE fBinary(const NN_OUTPUT_SHAPE &inp);

#endif
