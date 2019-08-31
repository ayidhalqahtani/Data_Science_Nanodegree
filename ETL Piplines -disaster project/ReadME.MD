# Disaster Response Pipeline Project

This project is about Data Engineering and create data engineering piplines and machine learning pipline to classify the text to
classes using model. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
        
 - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/mlmodel.pkl`
        
        
2. Run the following command in the app's directory to run your web app.
    `python run.py`
    

3. Go to http://0.0.0.0:3001/




* in workspace folder there are three folders which contain: 
 - app folder you can see run.py
 - data folder contains data and process_data.py file
 - models contains save model and train_classifier.py file 
 
 
