#ifndef FUNCTIONS_H
#define FUNCTIONS_H

std::vector<std::pair<std::vector<uint8_t>, uint8_t>> readCSV(const std::string& filepath);

void display_matrix(const std::vector<uint8_t>& data, const uint8_t size_a, const uint8_t size_b);

void show_dataset_element(const std::pair<std::vector<uint8_t>, uint8_t> dataset_elem);

std::vector<std::vector<double>> get_random_matrix(const int a, const int b);

std::vector<std::vector<double>> get_trained_model(std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_train);

#endif
