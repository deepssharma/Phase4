# Image Classification with Neural Networks
## Business Problem
* **Piedmont group, is looking for hiring a Data Scientist who can build a realistic model to efficiently screen the chest X-rays in pediatric patients that have pneumonia. 
They have certain remote locations where Radiology Department has only one Radiologist and so they often have troubles when Radiologist is out for certain reasons**
* Piedmont wants to avoid any lag in patient care and safety, minimize the diagnosis time and faster treatment timelines, and decrease the workload for Radiologist.
* My job is to build a Neural Network that detects the presence of pneumonia in X-ray images. I need to predict the status of the lungs (Normal vs pneumonia) as accurately as possible
while maximizing recall, i.e. identify majority of the **True Positive cases correctly** so that we catch as many kids with pneumonia as possible.
## Dataset Information
This dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) and it contains 5,856 validated Chest X-Ray images. The images are split into a training set and a testing set of independent patients.
Images are labeled as (disease:NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.
All chest X-ray imaging was performed as part of patients’ routine clinical care.
## Data Preparation
* The data came split into train and test folders, with training set containing 89% of the total data and test folder containg the remaining 11% of the data
* We split the training dataset into a validation set (10%) and kept 79% for training.
- Artificial Neural Network ANN 
- Convolutional Neural Network 
- Transfer Learning with VGG16
- Transfer Learning with ResNest50V2
