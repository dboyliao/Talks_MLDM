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
        
        let structure:[Int] = [2, 5, 6, 5, 10]
        self.cppWrapper = WrapperNN(structure: structure, lengthOfStructure: Int32(structure.count))
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

