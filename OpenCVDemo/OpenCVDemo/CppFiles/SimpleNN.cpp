//
//  SimpleNN.cpp
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//

#include "SimpleNN.hpp"

using namespace std;
using namespace cv;

SimpleNN::SimpleNN(string modelPath){
    
    this->load(modelPath);
    
    this->learning_rate = 0.1;
    this->random_range = 0.1;
    this->num_iteration = 1000;
}

SimpleNN::SimpleNN(vector<int> &nnStructure){
    this->structure = nnStructure;
    this->layers.reserve(nnStructure.size());
    this->weights.reserve(nnStructure.size()-1);
    
    // setup input layer
    int input_size = nnStructure[0];
    {
        Mat_<double> input_layer(input_size + 1, 1, 0.0f);
        input_layer(0, 0) = 1;
        this->layers.push_back(input_layer);
    }
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
    this->num_iteration = 1000;
}

void SimpleNN::load(string modelfile){
    
    string descript_str;
    int num_layers, layer_size;
    vector<int> network_structure;
    
    ifstream model_stream(modelfile);
    model_stream >> descript_str;
    cout << descript_str << endl; // NetworkStructure:
    
    for (int i = 0; i < num_layers; ++i){
        model_stream >> layer_size;
        network_structure.push_back(layer_size);
    }
    
    model_stream >> descript_str;
    cout << descript_str << endl; // NumberOfLayers:
    model_stream >> num_layers;
    network_structure.reserve(num_layers);
    
    
    this->weights.clear();
    
    for (int layer_index = 1; layer_index < num_layers; ++layer_index){
        int m = network_structure[layer_index -1] + 1; // add one for the bias term.
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
    cout << "loading complete" << endl;
}

void SimpleNN::save(string modelfile){
    ofstream model_stream(modelfile);
    
    model_stream << "NetworkStructure: ";
    for (int index = 0; index < this->structure.size(); ++index){
        model_stream << this->structure[index] << " ";
    }
    model_stream << endl;
    
    model_stream << "NumberOfLayers: ";
    model_stream << this->structure.size() << endl;
    
    for (int weight_index = 0; weight_index < this->weights.size(); ++weight_index){
        Mat_<double> weight = this->weights[weight_index].reshape(0, 1);
        for (int row_index = 0; row_index < weight.rows; ++row_index){
            model_stream << weight(row_index, 1) << " ";
        }
        model_stream << endl;
    }
}

void SimpleNN::setLearnParams(double learning_rate, double random_range, int num_iteration){
    
    this->learning_rate = learning_rate;
    this->random_range = random_range;
    this->num_iteration = num_iteration;
}

vector<int> SimpleNN::get_structure() const {
    return this->structure;
}

bool SimpleNN::train(const Mat_<double> &train_X,
                     const Mat_<double> &train_Y,
                     string &err_msg)
{
    /*
     Parameters
     ----------
     - Mat_<double> train_X: the training input data. It should be a NxM matrix
                             where N is the number of training data, M is the
                             size of input layer.
     - Mat_<double> train_Y: the target training output data. It should be a NxO 
                             matrix where N is the number of training data, O is 
                             the size of output layer.
     - string err_msg: error message (if any). It will be "" if there is no error.
     */
    
    int N = train_X.rows;
    
    if (train_Y.rows != N){
        err_msg = "the number of training input data does not match the number of output data.";
        return false;
    }
    
    srand((unsigned int) time(NULL));
    vector<Mat_<double> > deltas(this->weights.size());
    
    for (int iter_index = 0; iter_index < this->num_iteration; ++iter_index){
        if (iter_index % 100 == 0 && iter_index > 0){
            cout << "Iteration " << iter_index << " over total iteration " << this->num_iteration << endl;
        }
        
        // back-propagation with Stochastic Gradient Descend.
        int rand_index = rand() % N;
        Mat_<double> one_sample = train_X.row(rand_index).reshape(0, 1); // make it a column vector.
        
        Mat_<double> output;
        string err_msg;
        if (!this->predict(one_sample, output, err_msg)){
            cerr << err_msg << endl;
            return false;
        } else {
            deltas[this->weights.size() - 1] = 2.0*(output - this->layers[this->weights.size()-1]);
        }
    }
    
    err_msg = "";
    
    return true;
}

bool SimpleNN::predict(const Mat_<double> &test_X, Mat_<double> &result, string &err_msg){
    
    Mat_<double> input_data = test_X.reshape(0, 1);
    
    if (input_data.rows != this->structure[0]){
        err_msg = "wrong input size";
        return false;
    }
    
    this->layers[0] = input_data;
    int num_layers = (int) this->layers.size();
    
    for (int layer_id = 0; layer_id < num_layers - 1; ++layer_id){
        this->layers[layer_id+1] = tanh(this->weights[layer_id]*this->layers[layer_id]);
    }
    
    result = Mat_<double>(this->layers[num_layers-1].rows - 1, 1, 0.0f);
    for (int row_index = 0; row_index < result.rows; ++row_index){
        result(row_index, 0) = this->layers[num_layers - 1](row_index+1, 0);
    }
    err_msg = "";
    
    return true;
}

Mat_<double> tanh(Mat_<double> inputMat){
    Mat_<double> expx, expmx;
    cv::exp(inputMat, expx);
    cv::exp(-1.0*inputMat, expmx);
    
    Mat_<double> result(inputMat.rows, inputMat.cols, 0.0f);
    cv::divide(expx - expmx, expx + expmx, result);
    return result;
}

ostream& operator >> (ostream &os, const SimpleNN &nn){
    os << "<SimpleNN at " << &nn << " > ";
    
    vector<int> structure = nn.get_structure();
    for (int index = 0; index < structure.size()-1; ++index){
        os << structure[index] << "x";
    }
    
    os << structure[structure.size()-1] << endl;
    
    return os;
}
