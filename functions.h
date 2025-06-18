#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "config.h"

bool parseLine(const std::string &line, DataPoint &outDataPoint);
std::vector<DataPoint> readDataset(const std::string &filepath);

bool parseWeightLine(const std::string &line, size_t row, std::array<double, NN_INPUT_SIZE> &outRow);
WEIGHT_SHAPE readWeights();

void saveWeights(const WEIGHT_SHAPE &model, const std::string &filepath);

void shuffleDataset(std::vector<DataPoint> &dataset);

void displayConfusionMatrix(const std::array<std::array<double, NN_OUTPUT_SIZE>, NN_OUTPUT_SIZE> &data);

NeuralNetwork getTrainedModel(std::vector<DataPoint> &dataset_train);
NN_OUTPUT_SHAPE multiplyInputVectorWithWeights(const IMAGE_SHAPE &input_data, const WEIGHT_SHAPE &weights);
double evaluateModel(const NeuralNetwork &model, const std::vector<DataPoint> &dataset);

WEIGHT_SHAPE getRandomMatrix();
int indexOfMax(const NN_OUTPUT_SHAPE &output);

NN_OUTPUT_SHAPE sigmoid(const NN_OUTPUT_SHAPE &inp);
#endif
