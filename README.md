# Udacity-project2-Disaster_Response
Udacity, Data Science nanodegree, Project 2, Disaster Response

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Executing programm](#executing)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation">
 
Python versions 3.*.

Libraries used:
1. numpy
2. pandas
3. sklearn
4. nltk
5. re
6. sqlalchemy, sqlite3
7. pickle 

## Project Motivation <a name="motivation"></a>
This project is the second project of a Data Scientist Nanodegree Program by Udacity. In the project I used NLP and ML pipelines to build an API that classifies disaster messages.

## File Descriptions <a name="files"></a>
1. ScreenShots folder: executing program results (process_data.py, train_classifier.py). ML model screenshots results: graphs and an example message classification.
2. App folder-> run.py file: code of descriptive graphs of a dataset
3. data folder -> process_data.py: code loads, megres, cleans and saves csv files into SQLite database.
4. models folder -> train_classifier.py: code load cleaned data, builds, trains evaluates and saves ML model as a pickle file
5. ETL Pipeline Preparation.ipynb, ML Pipeline Preparation.ipynb: code drafts used to create process_data.py and train_classifier.py codes.

## Results <a name="results"></a>
The main findings of the API can be found in screenshots folder.

## Executing programm <a name="executing"></a>
1. To run ETL pipeline that cleans data and stores in database 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
2. To run ML pipeline that trains classifier and saves 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Code templates and data provided by Udacity. The data was originally sourced from Figure Eight.
Author of the modified codes: Grazyna Ostrowska
