#include <iostream>
#include <vector>
#include <cmath>

class SimplePatternPredictor {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;

public:
    SimplePatternPredictor() : weights({0.0, 0.0}), bias(0.0), learning_rate(0.01) {}

    double predict(const std::vector<double>& input) {
        double sum = bias;
        for (size_t i = 0; i < input.size(); i++) {
            sum += input[i] * weights[i];
        }
        return sum;
    }

    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<double>& outputs, 
               int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                double prediction = predict(inputs[i]);
                double error = outputs[i] - prediction;                
                
                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learning_rate * error * inputs[i][j];
                }
                bias += learning_rate * error;
            }
        }
    }

    void printWeights() {
        std::cout << "Pesos encontrados:" << std::endl;
        std::cout << "w1: " << weights[0] << ", w2: " << weights[1] << std::endl;
        std::cout << "bias: " << bias << std::endl;
    }
};

int main() {
    /* Dados de treinamento */
    std::vector<std::vector<double>> inputs = {
        {2, 3}, {1, 1}, {5, 2}, {12, 3}
    };
    std::vector<double> outputs = {10, 4, 14, 30};

    SimplePatternPredictor predictor;
    predictor.train(inputs, outputs, 10000);
    predictor.printWeights();

    /* testando infinito */
    while (true) {
        std::cout << "Digite dois numeros separados por espaco: ";
        std::vector<double> input(2);
        std::cin >> input[0] >> input[1];
        double prediction = predictor.predict(input);
        std::cout << "Previsao: " << prediction << std::endl;
    }
    /* Teste com novos dados */
    std::vector<double> test_input(2);
    std::cout << "\nTeste o modelo com novos números:\n";
    std::cout << "Digite dois números separados por espaço: ";
    std::cin >> test_input[0] >> test_input[1];

    double prediction = predictor.predict(test_input);
    std::cout << "Previsão: " << prediction << std::endl;

    return 0;
}