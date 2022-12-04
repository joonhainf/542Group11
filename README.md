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



