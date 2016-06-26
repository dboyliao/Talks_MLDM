//
//  ViewController.swift
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//  Copyright © 2016 spe3d. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    var cppWrapper:WrapperNN?

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        self.cppWrapper = WrapperNN(modelPath: "nn.model");
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

