# Steam Video Games: Attempting to predict discounts and genres. BZAN 542

This repo contains examples of how one can scrape a dataset for information using SVM, KNN, and Logistic Regression. This is a project for BZAN 542.

# Contacts
Luke Hwang

# Overview

This repo uses the "All 55,000 Games on Steam" dataset updated last at November 2022. The machine learning problem is to predict the availability of a discount using the quantitative variables in the dataset, and the game genre as a bonus.

The following methods were utilized:
-SVM
-KNN
-Logistic Regression

# Data Processing

The dataset used was admittedly from Kaggle https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022, however it was not without its challenges.
The easiest step was to figure out how to delimit the excel file. We delimited based on the semicolon, or ";". However there was also the challenge of appending an extra column to the dataset, which involved some conditional programming. We also needed to find out how to detect certain genres in games, which required some skills in text parsing. This was done in later versions of analysis_SVM.

# Support Vector Classification

This was the primary machine learning technique I used to try and figure out the problem. These files are all labeled analysis_SVM.py, analysis_SVM_2.py, etc.
I decided to use SVC in an attempt to seperate certain characteristics as predictive of the answer. I used a training set of 80% of the dataset and the rest of it was test.
I have plotted out SVM scatterplots in the respective files.
However, all SVM models fail to split the dataset in a meaningful way. In analysis_SVM.py, I try to find train the data on Negative Reviews and Initial Price to see if the games with discounts would be seperated this way. However, this did not work.

In another file I run that model that uses positive reviews and CCU (number of concurrent gamers) to predict reputation. Reputation would be defined as good if the ratio of positive to negative reviews is 10 or more. It would be bad otherwise.

And in my third SVM file I use positive reviews and CCU to predict if a game is free to play. A game is free to play if it has this tag in their genre's column in the dataset.





# K nearest Neighbors

I was able to generate satisfactory results using KNN. In the following results from analysis_KNN.py, I attempt to use 4 variables, {Positive Reviews, Negative Reviews, Initial Price, CCU} to predict the fifth variable {Has_Discount}. I predicted the absense of a discount with an f1 score of 98%.

              precision    recall  f1-score   support

           0       0.96      1.00      0.98     10643
           1       0.20      0.00      0.00       496

    accuracy                           0.96     11139
   macro avg       0.58      0.50      0.49     11139
weighted avg       0.92      0.96      0.93     11139

Figure 1: Output from classification report.

However, I think something is not right with the KNN model, so I hesitate to say this is a definitive model.


# Logistic Regression

Useful for plotting heatmaps, however I found its uses to be limited and I would primarily stick to SVMs.

# What to Run





