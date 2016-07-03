//
//  WrapperNN.h
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface WrapperNN : NSObject

-(instancetype) initWithModelPath: (NSString *) model;
-(int) predict: (UIImage *) inputImage;
-(void) debug;

@end
