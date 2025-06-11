#ifndef FUNCTIONS_H
#define FUNCTIONS_H

struct DataPoint {
    std::vector<uint8_t> pixels;
    uint8_t label;
};

std::vector<DataPoint> readDataset(const std::string &filepath);

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> readWeights();

void displayMatrix(const std::vector<uint8_t> &data, uint8_t size_a, uint8_t size_b);

void showDatasetElement(DataPoint dataset_elem);

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getRandomMatrix();

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getTrainedModel(std::vector<DataPoint> &dataset_train);

void saveWeights(const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &model, const std::string &filepath);

double evaluateModel(const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights, const std::vector<DataPoint> &dataset);

int getPrediction(const std::vector<uint8_t> &input_data, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights);

std::array<double, NN_OUTPUT_SIZE> multiplyInputVectorWithWeights(const std::vector<uint8_t> &input_data, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &weights);

int indexOfMax(const std::array<double, NN_OUTPUT_SIZE> &output);

void shuffleDataset(std::vector<DataPoint> &dataset);

void batch(int current_batch_index, const std::vector<DataPoint> &dataset_train, std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights);

std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> getDeltaMatrix(int start_index, int end_index, const std::vector<DataPoint> &dataset_train, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights);
void adjustWeights(std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &current_weights, const std::array<std::array<double, NN_INPUT_SIZE>, NN_OUTPUT_SIZE> &delta_matrix);

std::array<double, NN_OUTPUT_SIZE> biggest1Else0(const std::array<double, NN_OUTPUT_SIZE> &inp);
std::array<double, NN_OUTPUT_SIZE> sigmoid(const std::array<double, NN_OUTPUT_SIZE> &inp);
std::array<double, NN_OUTPUT_SIZE> fBinary(const std::array<double, NN_OUTPUT_SIZE> &inp);

#endif
