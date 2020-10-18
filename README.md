## Case1-Group2 - Smoking-Status-Detection
### I. Introduction
###### A. Background
The smoking status was extracted from clinical notes using the Natural Language Processing application embedded in the i2b2 NLP cell and called HITEx. The smoking status is mentioned in several different clinical notes such as the discharge summary or the history and physical exam report. The latter cites the smoking status the most frequently, and generally in more details. This project therefore chose to extract the smoking status from physical exam reports. The latter was then used to analyze these reports, and we used it in batch mode with the default configuration for smoking status extraction. The unstructured data of sentences that mentioned the smoking status of the patient, with the corresponding extracted smoking status (CURRENT SMOKER, PAST SMOKER, Non-Smoker, or unknown) and details about the location of the sentence, the reports, etc.

###### B. Problems
To let the computer to understand the human languages, the only method we can use is Natural Language Processing (NLP). The most common application for NLP in medical field is analyzing for medical records. Medical records contain lots of synonym and proper nouns which isn’t easy for analysis. In this project, we need to find the target vocabulary in this medical records that is helpful for smoking classification.

###### C. Goals
* Classify the smoking data to 4 smoking categories.
* Compare different kinds of machine learning model.
* Compare the 3 multi-classification methods – OvO, OvR, Multi-classification.


### II. Requirements
Python 3.6.8 or later with all requirements.txt dependencies installed, including. To install run:
```js
$ pip install -r requirements.txt
```

### III. Data Preprocessing and Feature Selection
There are three files about data preprocessing and feature selection
* **Feature_Selection_Function.py**
  * The file contain TFID and NGRAM function that will be used in the **Step2_Output_Result_main_code** and **Step1-4_CV_for_OvO_OvR_Multiclass**.
* **Preprocessing_Function.py**
  * The file contain 4 types of data preprocessing function that will be used in the **Step2_Output_Result_main_code** and **Step1-4_CV_for_OvO_OvR_Multiclass**..
* **TFIDF_with_Preprocessing.py**
  * The file data preprocessing and TFIDF code. It will output the numpy form data and will be input to the **Step1-1_CV_for_Decision_Tree**, **Step1-2_CV_for_Gradient_Boosting**, and **Step1-3_CV_for_Logistic_Regression**.


### IV. 
