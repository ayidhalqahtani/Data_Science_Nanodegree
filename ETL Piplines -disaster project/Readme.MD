# Disaster Response Pipeline Project

1. Project Overview: 

This project is about Data Engineering and create data engineering piplines and machine learning pipline to build a model for API that classifies Disaster messages into classes.

2.Project contents : 
   ETL Piplines: 
   It contains process_data.py file which load messages and categories dataset and then clean data and prepare it for the model.
   ML Piplines:
   It contains train_classifier.py file which load the database sqlite and then  build , train , test and extract the mdoel and save it to be used in run.py.
   Flask Web App:
   Here, where the user write the messages and the app will classify them into categories and here are screeshots of the app
   
   
   ![app screenshot](https://github.com/ayidhalqahtani/Data_Science_Nanodegree/blob/master/ETL%20Piplines%20-disaster%20project/Screen%20Shot%201441-01-01%20at%201.56.50%20AM.png)
   
   

![second example](https://github.com/ayidhalqahtani/Data_Science_Nanodegree/blob/master/ETL%20Piplines%20-disaster%20project/Screen%20Shot%201441-01-01%20at%201.53.03%20AM.png)





![third figure:](https://github.com/ayidhalqahtani/Data_Science_Nanodegree/blob/master/ETL%20Piplines%20-disaster%20project/VISULIZATION.png)



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
 - app folder you can see run.py to run the app 
 - data folder contains data and process_data.py file  to clean data and prepare it for the model.
 - models contains save model and train_classifier.py file to build , train , test and save the mdoel.
 
 
