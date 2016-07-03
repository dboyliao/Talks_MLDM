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
    
    ifstream model(modelfile);
    
    string title_str;
    int num_of_layers, layer_size;
    
    model >> title_str;
    cout << title_str << endl; // NumberOfLayers:
    model >> num_of_layers;
    cout << "num_of_layers: " << num_of_layers << endl;
    
    this->structure.clear();
    this->structure.reserve(num_of_layers);
    
    model >> title_str;
    cout << title_str << endl; // NetworkStructure:
    
    for (int i = 0; i < num_of_layers; ++i){
        model>> layer_size;
        cout << i << "-th layer size: " << layer_size << endl;
        this->structure.push_back(layer_size);
    }
    
    // setup layers (except output layer)
    this->layers.clear();
    for (int layer_id = 0; layer_id < num_of_layers - 1; ++layer_id){
        int n = this->structure[layer_id] + 1;
        Mat_<double> one_layer(n, 1, 0.0f);
        this->layers.push_back(one_layer);
    }
    
    // setup output layer
    {
        int n = this->structure[num_of_layers-1];
        Mat_<double> output_layer(n, 1, 0.0f);
        this->layers.push_back(output_layer);
    }
    
    this->weights.clear();
    this->weights.reserve(num_of_layers - 1);
    
    for (int layer_id = 1; layer_id < num_of_layers; ++layer_id){
        int m = this->structure[layer_id-1] + 1;
        int n = this->structure[layer_id];
        
        Mat_<double> weight(n, m, 0.0f);
        for (int row_index = 0; row_index < n; ++row_index){
            for (int col_index = 0; col_index < m; ++col_index){
                double w;
                model >> w;
                weight(row_index, col_index) = w;
            }
        }
        cout << "weight shape: " << weight.rows << "x" << weight.cols << endl;
        this->weights.push_back(weight);
    }
    
    model.close();
    
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

// MARK: implement not complete yet.
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
        Mat_<double> one_target = train_Y.row(rand_index).reshape(0, 1);
        
        Mat_<double> output;
        string err_msg;
        if (!this->predict(one_sample, output, err_msg)){
            cerr << err_msg << endl;
            return false;
        } else {
            deltas[this->weights.size() - 1] = 2.0*(one_target - output);
        }
    }
    
    err_msg = "";
    
    return true;
}

bool SimpleNN::predict(const Mat_<double> &test_X, Mat_<double> &result, string &err_msg){
    
    Mat_<double> input_data = test_X.reshape(0, test_X.rows*test_X.cols); // make it column vector
    
    if (input_data.rows != this->structure[0]){
        err_msg = "wrong input size";
        return false;
    }
    
    for (int row_index = 1; row_index < this->layers[0].rows; ++row_index){
        this->layers[0](row_index, 0) = input_data(row_index-1, 0);
    }

    int num_layers = (int) this->layers.size();
    
    for (int layer_id = 0; layer_id < num_layers - 2; ++layer_id){
        Mat_<double> product = tanh(this->weights[layer_id]*this->layers[layer_id]);
        
        for (int row_index = 1; row_index < this->layers[layer_id+1].rows; ++row_index){
            this->layers[layer_id+1](row_index, 0) = product(row_index-1, 0);
        }
    }
    
    // compute the output layer
    {
        int layer_id = num_layers - 2;
        this->layers[layer_id + 1] = tanh(this->weights[layer_id] * this->layers[layer_id]);
    }
    
    result = this->layers[num_layers - 1]; // return last layers (output layer).
    cout << "result:\n" << result << endl;
    
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

vector<int> SimpleNN::get_structure() const {
    return this->structure;
}

vector<Mat_<double> > SimpleNN::get_weights() const {
    return this->weights;
}

vector<Mat_<double> > SimpleNN::get_layers() const {
    return this->layers;
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
