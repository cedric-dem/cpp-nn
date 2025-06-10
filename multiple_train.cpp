#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"
#include "functions.h"

void showStatistics(const std::vector<double> &data) {
    if (data.empty()) {
        std::cout << "Vector is empty." << std::endl;
        return;
    }

    auto [minIt, maxIt] = std::minmax_element(data.begin(), data.end());
    double min = *minIt;
    double max = *maxIt;

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double average = sum / data.size();

    std::vector<double> sorted = data; // Make a copy to sort
    std::sort(sorted.begin(), sorted.end());
    double median;
    size_t n = sorted.size();
    if (n % 2 == 0) {
        median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    } else {
        median = sorted[n / 2];
    }
    std::cout << "Minimum: " << min << std::endl;
    std::cout << "Average: " << average << std::endl;
    std::cout << "Median:  " << median << std::endl;
    std::cout << "Maximum: " << max << std::endl;
}

int main() {

    std::cout << "========================================================> Begin to load datasets" << std::endl;
    std::cout << "====> Begin to load dataset train" << std::endl;
    std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_train = readDataset(DATASET_TRAIN_PATH);
    std::cout << "Finished loading train set, size " << dataset_train.size() << " . " << std::endl;

    std::cout << "====> Begin to load dataset test" << std::endl;
    const std::vector<std::pair<std::vector<uint8_t>, uint8_t>> dataset_test = readDataset(DATASET_TEST_PATH);
    std::cout << "=> Finished loading test set, size " << dataset_test.size() << " . " << std::endl;

    std::vector<double> history_scores(EXECUTIONS_QUANTITY, 0);

    std::cout << "========================================================> Begin to train and test" << std::endl;
    for (int current_train = 0; current_train < EXECUTIONS_QUANTITY; ++current_train) {
        std::cout << "========> Iteration : " << current_train << std::endl;

        std::cout << "====> Begin to train the model" << std::endl;
        const std::vector<std::vector<double>> model = get_trained_model(dataset_train);

        std::cout << "====> Begin evaluation on the test set" << std::endl;
        history_scores[current_train] = evaluate_model(model, dataset_test);
    }

    std::cout << "========================================================> Finished exec, results : "<< std::endl;
    showStatistics(history_scores);

    return 0;
}
