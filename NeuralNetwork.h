#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "DataPoint.h"
#include "config.h"
#include <vector>

class NeuralNetwork {
  private:
    WEIGHT_SHAPE weights;

  public:
    explicit NeuralNetwork(bool load_weights);

    void doOneBatch(size_t current_batch_index, const std::vector<DataPoint> &dataset_train);

    WEIGHT_SHAPE getDeltaMatrix(size_t start_index, size_t end_index, const std::vector<DataPoint> &dataset_train) const;

    void adjustWeights(const WEIGHT_SHAPE &delta_matrix);

    WEIGHT_SHAPE getWeights() const;

    int getPrediction(const IMAGE_SHAPE &input_data) const;

    NN_OUTPUT_SHAPE multiplyInputVectorWithWeights(const IMAGE_SHAPE &input_data) const;
};

#endif // NEURALNETWORK_H