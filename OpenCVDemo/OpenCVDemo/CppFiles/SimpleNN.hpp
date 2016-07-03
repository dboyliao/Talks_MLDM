//
//  SimpleNN.hpp
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
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
    void setLearnParams(double learning_rate, double random_range, int num_iteration);
    bool train(const Mat_<double> &train_X,
               const Mat_<double> &train_Y,
               string &err_msg);
    bool predict(const Mat_<double> &test_X, Mat_<double> &result, string &err_msg);
    void load(string modelfile);
    void save(string modelfile);
    vector<int> get_structure() const;
    vector<Mat_<double> > get_weights() const;
    vector<Mat_<double> > get_layers() const;

private:
    double learning_rate;
    double random_range;
    int num_iteration;
    vector<Mat_<double> > weights;
    vector<Mat_<double> > layers;
    vector<int> structure;
};

Mat_<double> tanh(Mat_<double> inputMat);
ostream& operator >>(ostream &os, const SimpleNN &nn);

#endif /* SimpleNN_hpp */
