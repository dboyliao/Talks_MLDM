//
//  ViewController.swift
//  OpenCVDemo
//
//  Created by DboyLiao on 6/26/16.
//

import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView:UIImageView!
    @IBOutlet weak var textField:UITextField!
    @IBOutlet weak var pickButton:UIButton!
    @IBOutlet weak var predictButton:UIButton!
    
    var cppWrapper:WrapperNN!

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
    
    @IBAction func pressButton(sender:UIButton!){
        
        switch sender {
        case self.pickButton:
            let nextVC = UIImagePickerController()
            nextVC.sourceType = .PhotoLibrary
            nextVC.delegate = self
            self.presentViewController(nextVC, animated: true, completion: nil)
            
        case self.predictButton:
            if let image = self.imageView.image, wrapper = self.cppWrapper {
                print("predict! (nn)")
                let i = wrapper.predict(image)
                
                if i == 0 {
                    self.textField.text = "It's a 0"
                } else {
                    self.textField.text = "It's a 1"
                }
            }
        default:
            break
        }
        
    }
    
    func imagePickerController(picker: UIImagePickerController, didFinishPickingImage image: UIImage, editingInfo: [String : AnyObject]?) {
        
        self.imageView.image = image;
        picker.dismissViewControllerAnimated(true, completion: nil)
    }

}

