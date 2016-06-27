//
//  SimpleNN.hpp
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//  Copyright © 2016 spe3d. All rights reserved.
//

#ifndef SimpleNN_hpp
#define SimpleNN_hpp

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <time.h> // time
#include <stdlib.h> // srand, rand
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

class SimpleNN {
public:
    SimpleNN(string modelPath);
    SimpleNN(vector<int> &nnStructure);
    
public:
    void setLearnParams(double learning_rate, double random_range);
    void train(const Mat_<double> &train_X, const Mat_<double> &train_Y);
    Mat_<double> predict(const Mat_<double> &test_X);
    void load(string model);
    void save(ofstream &fstream);
    vector<int> get_structure() const;
    
private:
    double learning_rate;
    double random_range;
    vector<Mat_<double> > weights;
    vector<Mat_<double> > layers;
    vector<int> structure;
};

ostream& operator >>(ostream &os, const SimpleNN &nn);

#endif /* SimpleNN_hpp */
