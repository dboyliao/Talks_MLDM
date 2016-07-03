//
//  ViewController.swift
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//

import UIKit

class ViewController: UIViewController {
    
    var cppWrapper:WrapperNN?

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        print("loading model.")
        if let model_path = NSBundle.mainBundle().pathForResource("model", ofType: "txt") {
            print("model file found: \(model_path)")
            self.cppWrapper = WrapperNN(modelPath: model_path)
        } else {
            print("model file not found....")
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

