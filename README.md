# Image Classification with Neural Networks
## Business Problem
---------------------------------
* **Piedmont group, is looking for hiring a Data Scientist who can build a realistic model to efficiently screen the chest X-rays in pediatric patients that have pneumonia. 
They have certain remote locations where Radiology Department has only one Radiologist and so they often have troubles when Radiologist is out for certain reasons**
* Piedmont wants to avoid any lag in patient care and safety, minimize the diagnosis time and faster treatment timelines, and decrease the workload for Radiologist.
* My job is to build a Neural Network that detects the presence of pneumonia in X-ray images. I need to predict the status of the lungs (Normal vs pneumonia) as accurately as possible
while maximizing recall, i.e. identify majority of the **True Positive cases correctly** so that we catch as many kids with pneumonia as possible.
## Dataset Information
---------------------------------
This dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) and it contains 5,856 validated Chest X-Ray images. The images are split into a training set and a testing set of independent patients.
Images are labeled as (disease:NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.
All chest X-ray imaging was performed as part of patients’ routine clinical care.
## Data Preparation
---------------------------------
* The data came split into train and test folders, with training set containing 89% of the total data and test folder containg the remaining 11% of the data
* We split the training dataset into a validation set (9%) and kept 80% for training.
* All images were down-scaled to a size of 128 x 128 pixels.
* Pixels values (0-255) normalized to 0-1.
* A comparison of some images without any pre-processing and after pre-processing is shown below:

**Original Images:**
![Original Images](https://github.com/deepssharma/Phase4/blob/main/images/RawImages.png)
**Processed Images:**
![Pre-processed Images](https://github.com/deepssharma/Phase4/blob/main/images/ScaledImages.png)

## Modelling
----
This analysis used following models:
- Artificial Neural Networks, also known as Neural Nets)(ANN or NN).
- Convolutional Neural Networks (CNNs)
- Pre-trained Modules/Transfer Learning: We have used the following models;
  - Xception
  - RESNET101
## Evaluation
----
* We used **confusion matrix** with **Accuaracy** and **Recall** values on test dataset as the performance metric for training our models. In particular, we want to minimize  the false negatives for pneumonia class i.e. maximize True positives (Recall) as we dont want patients with pneumonia to be mis-diagonsed.

* The following two top models were picked up as the best models.
Shown below is the loass and accuracy trends for validation and training dataset. We do not see any overfitting and an overall accuracy of 96% and 94% was reached for these two models.
![CompareModels_train_val_acc](https://github.com/deepssharma/Phase4/blob/main/images/accuracy_top_two_models_comp.png)

* We reached an overall acuuracy of 91% and recall value of 99% for the test dataset.
![CompareModels_ConfusionMatrices](https://github.com/deepssharma/Phase4/blob/main/images/TopModels_CM.png)

**Deeper CNN trained on Augmented data (CNN_deep_aug)** was chosen as the final model since it gave the best performance on test dataset by missing only 3 pneumonia-positive cases out of 390, and 55 out of 234 normal cases. 

* The performance on the test set using **CNN_deep_aug**:
    - overall accuracy score of 91%, 
    - recall score of 99% for class pneumonia, and 76% for normal class,
    - f1 score of 93% for class pneumonia, and 86% for normal class.
<img width="596" alt="Confusion Matrix" src="https://github.com/deepssharma/Phase4/blob/main/images/classificatio_report.png">

## Features Visualization:
----
* Below is the visualization of a sample channel for each of the activation layers from CNN architecture. 
* The first layers of CNN learn detailed represenations while the outer layers learn more abstract patterns. 
![ActivationChannels](https://github.com/deepssharma/Phase4/blob/main/images/features.png)

## Models Visualization using LIME:
-----
**LIME**, the acronym for local interpretable model-agnostic explanations, is a technique that approximates any black box machine learning model with a local, interpretable model to explain each individual prediction (https://arxiv.org/abs/1602.04938)
https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/.

* Lets visualise one image using LIME and see how the given model has made demarcations to label the image.

**Mask Boundaries for two models:** Towards and against - green and red respectively.
![Masked](https://github.com/deepssharma/Phase4/blob/main/images/Topmodels_lime_masks.png)

**Heatmap** - The more blue it is, the higher positive impact!
![Boundaries](https://github.com/deepssharma/Phase4/blob/main/images/Topmodels_lime_comp.png)

* The two models work differently as we can see that they have slighlty different masks and boundaries.
## Recommendations
----
* Neural network are a useful tool to aid the healthcare professional in stream-lining the diagnosing process when classifying x-ray images. This will allow for a quicker return time and greater patient satisfaction.

* Detecting children that have pneumonia as early as possible is very important for early intervention. Using neural networks for x-ray image clasiification will significantly reduce the waiting times for patients to hear back from radiologist and their treatment recommendations.

* The Radiologists will have reduction in his work-load and there will be a mechanism to provide continued care for patients when the radiologist is out on sick leave or has to take vacations.

## Further Improvements
----
* We probably need to implememt weights in the training to take into account class imbalance using oversampling techniques which could improve performance.
* Cropping images to remove unnecessary captured details such as R may bring some improvement to models.
* Also make it a multilabel-classification problem where we identify the pneumonias as virus or bacteria related. The treatment plan differs for the two categories.
## Repository Structure
 ------
    ├── images                              Images folder, containing all referenced image files
    ├── project4_image_classification.ipynb Main Jupyter notebook, contains analysis
    ├── Notebook.pdf                        PDF version of main Jupyter notebook
    ├── Presentation.pdf                    PDF Version of project presentation                                        
    └── README.md                           The top-level README  
## Contact Info:
-----
* Email: deeps.sharma@gmail.com
* GitHub: [@deepssharma](https://github.com/deepssharma)
* [LinkedIn](https://www.linkedin.com/in/deepali-sharma-a83a126/) 
