# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:55:55 2021

@author: Salehin
"""
import joblib
import pandas as pd

if __name__ == "__main__":
    user_input = input("Enter the path of your trained model: ")
    #check if the model can be loaded. Otherwise throw a filenotfound error
    try:
        model=joblib.load(user_input)
    except FileNotFoundError:
        print("File not found")
    #Because the number of input features are not specified I will assume that I have the dictionary of new attributes 
    #available to me. I name the dictionary as new_features I just need to convert it to a dataframe
    attribute_data=pd.DataFrame(new_features)
    output=model.predict(attribute_data)
    print(output)
    
    
    