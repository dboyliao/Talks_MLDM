//
//  SimpleNN.cpp
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//  Copyright © 2016 spe3d. All rights reserved.
//

#include "SimpleNN.hpp"

using namespace std;
using namespace cv;

SimpleNN::SimpleNN(){
    this->structure = vector<int>(0);
    this->weights = vector<Mat_<double> >(0);
    this->layers = vector<Mat_<double> >(0);
    
    this->learning_rate = 0.1;
    this->random_range = 0.1;
}

SimpleNN::SimpleNN(vector<int> &nnStructure){
    this->structure = nnStructure;
    this->layers.reserve(nnStructure.size());
    this->weights.reserve(nnStructure.size()-1);
    
    for (int layer_id = 0; layer_id < nnStructure.size(); ++layer_id){
        Mat_<double> one_layer = Mat_<double>(nnStructure[layer_id] + 1, 1, 0.0f);
        this->layers.push_back(one_layer);
    }
    
    srand((int)time(NULL)); // set seed;
    
    for (int layer_id = 0; layer_id < nnStructure.size()-1; ++layer_id){
        Mat_<double> weight = Mat_<double>(nnStructure[layer_id+1], nnStructure[layer_id], 0.0f);
        for (int row_id = 0; row_id < weight.rows; ++row_id){
            for (int col_id = 0; col_id < weight.cols; ++col_id){
                weight(row_id, col_id) = this->random_range * ((double) (rand()/RAND_MAX));
            }
        }
        this->weights.push_back(weight);
    }
    
    this->learning_rate = 0.1;
    this->random_range = 0.1;
}

void SimpleNN::load(string model){
    cout << model << endl;
}

void SimpleNN::setLearnParams(double learning_rate, double random_range){
    
    this->learning_rate = learning_rate;
    this->random_range = random_range;
}

vector<int> SimpleNN::get_structure() const {
    return this->structure;
}

ostream& operator >> (ostream &os, const SimpleNN &nn){
    os << "Simple Neural Network: ";
    
    vector<int> structure = nn.get_structure();
    for (int index = 0; index < structure.size()-1; ++index){
        os << structure[index] << "x";
    }
    
    os << structure[structure.size()-1] << endl;
    
    return os;
}
