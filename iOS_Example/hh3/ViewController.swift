//
//  ViewController.swift
//  hh3
//
//  Created by Marshall White on 10/1/17.
//  Copyright © 2017 Marshall White. All rights reserved.
//

import UIKit

// Simple UIViewController for accepting user input for features
// and displaying a house price prediction
class ViewController: UIViewController {
    
    @IBOutlet weak var bathroomField: UITextField!
    @IBOutlet weak var bedroomField: UITextField!
    @IBOutlet weak var sizeField: UITextField!
    @IBOutlet weak var predictionLabel: UILabel!
    
    /// Formatter for the price output
    let priceFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 0
        formatter.usesGroupingSeparator = true
        formatter.locale = Locale(identifier: "en_US")
        return formatter
    }()
    
    // Model for price prediction
    let model = linearhousingmodel()
    
    // Called when the user presses the 'Predict' button
    @IBAction func predictPressed(_ sender: Any) {
        
        // Simply dismiss the keyboard
        bathroomField.resignFirstResponder()
        bedroomField.resignFirstResponder()
        sizeField.resignFirstResponder()
        
        // Obtain the user input and attempt to coerce it into feature input
        guard let bathrooms = bathroomField.text,
            let bedrooms = bedroomField.text,
            let size = sizeField.text,
            let bathroomFeature = Double(bathrooms),
            let bedroomFeature = Double(bedrooms),
            let sizeFeature = Double(size)
            else { return }
        
        // Ask the model for a prediction
        guard let prediction = try? model.prediction(Bedrooms: bedroomFeature, Bathrooms: bathroomFeature, Size: sizeFeature) else { return }
        
        // Format and display the prediction in the UILabel
        predictionLabel.text = priceFormatter.string(for: prediction.Price)
    }
}

