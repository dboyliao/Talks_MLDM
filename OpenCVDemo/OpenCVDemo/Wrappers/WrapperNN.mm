//
//  WrapperNN.m
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//

#import "WrapperNN.h"
#include "SimpleNN.hpp"
#include <string>
#include "opencv2/imgcodecs/ios.h"

using namespace std;
using namespace cv;

@implementation NSString (Cpp)

- (std::string) cppStringFromNSString {
    
    std::string cppString = std::string([self cStringUsingEncoding:NSUTF8StringEncoding]);
    return cppString;
}

@end

@interface WrapperNN ()

@property SimpleNN* network;

@end

@implementation WrapperNN

-(instancetype) initWithModelPath: (NSString *) modelPath {
    
    self = [super init];
    
    if (self) {
        self.network = new SimpleNN([modelPath cppStringFromNSString]);
    }
    
    return self;
}

-(int) predict: (UIImage *)inputImage {
    Mat_<uchar> image_mat;
    Mat_<double> input, prediction;
    UIImageToMat(inputImage, image_mat);
    
    if (image_mat.channels() > 1){
        // convert to grayscale
        cvtColor(image_mat, image_mat, CV_BGR2GRAY);
    }
    
    input = Mat_<double>(image_mat);
    string err_msg;
    if (!self.network->predict(input, prediction, err_msg)){
        cerr << "prediciton fail" << endl;
        cerr << err_msg << endl;
    }
    
    if (prediction(0, 0) >= prediction(0, 1)){
        return 0;
    }
    
    return 1;
    
}

@end
