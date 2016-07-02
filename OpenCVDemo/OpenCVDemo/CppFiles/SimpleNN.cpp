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

SimpleNN::SimpleNN(string modelPath){
    
    this->load(modelPath);
    
    this->learning_rate = 0.1;
    this->random_range = 0.1;
}

SimpleNN::SimpleNN(vector<int> &nnStructure){
    this->structure = nnStructure;
    this->layers.reserve(nnStructure.size());
    this->weights.reserve(nnStructure.size()-1);
    
    // setup input layer
    int input_size = nnStructure[0];
    this->layers.push_back(Mat_<double>(input_size, 1, 0.0f));
    
    for (int layer_id = 1; layer_id < nnStructure.size(); ++layer_id){
        Mat_<double> one_layer = Mat_<double>(nnStructure[layer_id] + 1, 1, 0.0f);
        one_layer(0, 0) = 1;
        this->layers.push_back(one_layer);
    }
    
    srand((int)time(NULL)); // set seed;
    
    for (int layer_id = 0; layer_id < nnStructure.size()-1; ++layer_id){
        Mat_<double> weight = Mat_<double>(nnStructure[layer_id+1], nnStructure[layer_id], 0.0f);
        
        // initiate weight.
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

void SimpleNN::load(string modelfile){
    
    string descript_str;
    int num_layers, layer_size;
    vector<int> network_structure;
    
    ifstream model_stream(modelfile);
    model_stream >> descript_str;
    cout << descript_str << endl;
    
    model_stream >> num_layers;
    network_structure.reserve(num_layers);
    
    for (int i = 0; i < num_layers; ++i){
        model_stream >> layer_size;
        network_structure.push_back(layer_size);
    }
    
    this->weights.clear();
    
    for (int layer_index = 1; layer_index < num_layers; ++layer_index){
        int m = network_structure[layer_index -1];
        int n = network_structure[layer_index];
        Mat_<double> weight(m, n, 0.0f);
        double w;
        
        for (int row_index = 0; row_index < m; ++row_index){
            for (int col_index = 0; col_index < n; ++col_index){
                model_stream >> w;
                weight(row_index, col_index) = w;
            }
        }
        this->weights.push_back(weight);
    }
}

void SimpleNN::setLearnParams(double learning_rate, double random_range){
    
    this->learning_rate = learning_rate;
    this->random_range = random_range;
}

vector<int> SimpleNN::get_structure() const {
    return this->structure;
}

Mat_<double> SimpleNN::predict(const Mat_<double> &test_X){
    
    this->layers[0] = test_X;
    int num_layers = (int) this->layers.size();
    
    for (int layer_id = 0; layer_id < num_layers - 1; ++layer_id){
        this->layers[layer_id+1] = this->weights[layer_id]*this->layers[layer_id];
    }
    
    Mat_<double> result(this->layers[num_layers-1].rows-1, 0, 0.0f);
    for (int row_index = 0; row_index < result.rows; ++row_index){
        result(row_index, 0) = this->layers[num_layers - 1](row_index+1, 0);
    }
    
    return result;
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
