//
//  WrapperNN.m
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//  Copyright © 2016 spe3d. All rights reserved.
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

@property SimpleNN* cppInstance;

@end

@implementation WrapperNN

-(instancetype) initWithModelPath: (NSString *) modelPath {
    
    self = [super init];
    
    if (self) {
        self.cppInstance = new SimpleNN([modelPath cppStringFromNSString]);
    }
    
    return self;
}

-(int) predict: (UIImage *)inputImage {
    
    Mat_<uchar> image_mat;
    UIImageToMat(inputImage, image_mat);
    
    return 0;
    
}

@end
