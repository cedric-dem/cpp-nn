//
// Created by ced on 11/06/25.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "DataPoint.h"
#include "config.h"

class NeuralNetwork {

  private:
    WEIGHT_SHAPE weights;

  public:
    NeuralNetwork(const bool load_weights);
    void doOneBatch(const int current_batch_index, const std::vector<DataPoint> &dataset_train);
    WEIGHT_SHAPE getDeltaMatrix(const int start_index, const int end_index, const std::vector<DataPoint> &dataset_train) const;
    void adjustWeights(const WEIGHT_SHAPE &delta_matrix);
    WEIGHT_SHAPE getWeights() const;
};

#endif // NEURALNETWORK_H
