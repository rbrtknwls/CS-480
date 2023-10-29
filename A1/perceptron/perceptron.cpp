#include "perceptron.h"
#include <iostream>

bool Perceptron::classifyVal( int label, std::vector<double> *val ) {
    double sum = 0;
    for (unsigned int i = 0; i < val->size(); i++){ sum += val->at(i) * weights.at(i); }

    if (label*sum <= errorTerm) {
        std::cout << "-";
        for (unsigned int i = 0; i < val->size(); i++){

            weights[i] += label*val->at(i);

            //std::cout << weights[i] << std::endl;
        }
        return false;
    }
    std::cout << "+";
    for (unsigned int i = 0; i < val->size(); i++) {
        //weights[i] += label*val->at(i);
        std::cout << weights[i] << std::endl;
    }
    return true;

}

Perceptron::Perceptron( int d ) {
    b = 0;
    errorTerm = 0;

    for ( int i = 0; i < d; i++ ) {
        weights.push_back(0);
    }

}

int main() {
    Perceptron p = Perceptron(3);
    std::vector<double> val1 = {0.1, -0.01, 0.01};
    std::vector<double> val2 = {-0.01, 0.01, 0.1};
    std::vector<double> val3 = {0.1, 0.01, 0.01};

    for (int i = 0; i < 20; i++) {
        p.classifyVal(1, &val1);
        p.classifyVal(-1, &val2);
        p.classifyVal(1, &val3);
    }


}