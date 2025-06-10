#ifndef FUNCTIONS_H
#define FUNCTIONS_H

struct DataPoint {
    std::vector<uint8_t> pixels;
    uint8_t label;
};

std::vector<DataPoint> readDataset(const std::string &filepath);

std::vector<std::vector<double>> readWeights();

void displayMatrix(const std::vector<uint8_t> &data, uint8_t size_a, uint8_t size_b);

void showDatasetElement(DataPoint dataset_elem);

std::vector<std::vector<double>> getRandomMatrix(int a, int b);

std::vector<std::vector<double>> getTrainedModel(std::vector<DataPoint> &dataset_train);

void saveWeights(const std::vector<std::vector<double>> &model, const std::string &filepath);

double evaluateModel(const std::vector<std::vector<double>> &weights, const std::vector<DataPoint> &dataset);

int getPrediction(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights);

std::vector<double> multiplyInputVectorWithWeights(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights);

int indexOfMax(const std::vector<double> &output);

void shuffleDataset(std::vector<DataPoint> &dataset);

void batch(int current_batch_index, const std::vector<DataPoint> &dataset_train, std::vector<std::vector<double>> &current_weights);

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getDeltaMatrix(int start_index, int end_index, const std::vector<DataPoint> &dataset_train, const std::vector<std::vector<double>> &current_weights);
void adjustWeights(std::vector<std::vector<double>> &current_weights, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE>  &delta_matrix);

std::vector<double> biggest1Else0(const std::vector<double> &inp);
std::vector<double> sigmoid(const std::vector<double> &inp);
std::vector<double> fBinary(const std::vector<double> &inp);

#endif
