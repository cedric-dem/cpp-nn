#ifndef FUNCTIONS_H
#define FUNCTIONS_H

std::vector<std::pair<std::vector<uint8_t>, uint8_t>> readDataset(const std::string &filepath);

std::vector<std::vector<double>> readWeights();

void display_matrix(const std::vector<uint8_t> &data, uint8_t size_a, uint8_t size_b);

void show_dataset_element(std::pair<std::vector<uint8_t>, uint8_t> dataset_elem);

std::vector<std::vector<double>> get_random_matrix(int a, int b);

std::vector<std::vector<double>> get_trained_model(std::vector<std::pair<std::vector<uint8_t>, uint8_t>> &dataset_train);

void save_weights(const std::vector<std::vector<double>> &model, const std::string &filepath);

double evaluate_model(const std::vector<std::vector<double>> &weights, const std::vector<std::pair<std::vector<uint8_t>, uint8_t>> &dataset);

int get_prediction(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights);

std::vector<double> multiply_input_vector_with_weights(const std::vector<uint8_t> &input_data, const std::vector<std::vector<double>> &weights);

int index_of_max(const std::vector<double> &output);

void shuffle_dataset(std::vector<std::pair<std::vector<uint8_t>, uint8_t>> &dataset);

#endif
