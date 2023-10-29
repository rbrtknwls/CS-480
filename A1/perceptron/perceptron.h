#ifndef CS_480_PERCEPTRON_H
#define CS_480_PERCEPTRON_H
#include <iostream>
#include <vector>

class Perceptron {
    private:
        double errorTerm;
        double b;
        std::vector<double> weights;
    public:
        explicit Perceptron( int d );
        bool classifyVal( int label, std::vector<double> *val );

};


#endif //CS_480_PERCEPTRON_H
