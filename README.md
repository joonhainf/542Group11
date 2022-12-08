# Steam Video Games: Attempting to predict discounts and genres. BZAN 542

This repo contains examples of how one can scrape a dataset for information using SVM, KNN, and Logistic Regression. This is a project for BZAN 542.

# Contacts
Luke Hwang

# Overview

This repo uses the "All 55,000 Games on Steam" dataset updated last at November 2022. The machine learning problem is create a model. In the readme I will go through many attempted machine learning methods, and explain why I chose my final model and method. By the way, the final model is a logistic regression predicting Y = a game with the tags "FPS" and "Multiplayer" and x = Positive and Negative Reviews. However before this, I make many attempts with different methods including SVM and KNN.

The following methods were utilized:
- SVM
- KNN
- Logistic Regression

# Data Processing

The dataset used was admittedly from Kaggle https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022, however it was not without its challenges.
The easiest step was to figure out how to delimit the excel file. We delimited based on the semicolon, or ";". However there was also the challenge of appending an extra column to the dataset, which involved some conditional programming. We also needed to find out how to detect certain genres and tags in games, which required some skills in text parsing. This was done in later versions of analysis_SVM.

# Support Vector Classification

This was the primary machine learning technique I used to try and figure out the problem. These files are all labeled analysis_SVM.py, analysis_SVM_2.py, etc.
I decided to use SVC in an attempt to seperate certain characteristics as predictive of the answer. I used a training set of 80% of the dataset and the rest of it was test.
I have plotted out SVM scatterplots in the respective files.
However, all SVM models fail to split the dataset in a meaningful way. In analysis_SVM.py, I try to find train the data on Negative Reviews and Initial Price to see if the games with discounts would be seperated this way. However, this did not work.

![SVM1](https://user-images.githubusercontent.com/28285099/205476340-539e2e7a-8c8d-4762-80dc-114df117335f.png)

In another file I run that model that uses positive reviews and CCU (number of concurrent gamers) to predict reputation. Reputation would be defined as good if the ratio of positive to negative reviews is 10 or more. It would be bad otherwise.

![SVM2](https://user-images.githubusercontent.com/28285099/205476344-dcf60a4b-369c-437d-8a5e-f56c9419d980.png)

In my third SVM file I use positive reviews and CCU to predict if a game is free to play. A game is free to play if it has this tag in their genre's column in the dataset.

![SVM3](https://user-images.githubusercontent.com/28285099/205476353-2a0be4ee-e7cd-4e18-94c5-81178265ed40.png)

In my fourth SVM file I tried seperating the data by having a model with x = Positive and Negative values and Y = games with the tags "FPS" and "Multiplayer". The result of this was better than my previous 3 attempts, but still not great.

![SVM4](https://user-images.githubusercontent.com/28285099/205476400-655f4563-721d-4675-920f-7b7f7783b2cc.png)

I then realized that the Support Vector Classification model only works when there are extreme differences between two classes of variables. If there is simply a relationship between x and Y that is not enough. So, I turned to other methods. 



# K nearest Neighbors

I was able to generate satisfactory results using KNN. In the following results from analysis_KNN.py, I attempt to use 4 variables, {Positive Reviews, Negative Reviews, Initial Price, CCU} to predict the fifth variable {Has_Discount}. I predicted the absense of a discount with an f1 score of 98%.

              
![KNNclassification_report](https://user-images.githubusercontent.com/28285099/205476227-3691e64c-9306-4fd7-9973-d382890832dd.png)





However, I think something is not right with the KNN model, so I hesitate to say this is a definitive model.



# Logistic Regression

I ultimately decided to use Logistic Regression to create a model for this project. Logistic regression is good for data that is right skewed like this, which is a big reason I prefer it over linear regression. The final model is a logistic regression predicting Y = a game with the tags "FPS" and "Multiplayer" and x = Positive and Negative Reviews. One important thing I needed to do was to append a new column to the dataset that is a binary variable signifying whether a game is a multiplayer fps. So I converted the "Tags" column to a string and used a conditional to parse this data. I then put this data into a new column called "big_FPS".

I then split the data into 75% train 25% test. I run logistic regression, fitting the train data and testing it on test data. I get a score of around 0.9824. I also create a confusion matrix and put this into a heatmap.

![LOG3](https://user-images.githubusercontent.com/28285099/205476578-af76934f-0fcb-4718-a954-e2287d58d9e6.png)

Here is the classification report.

![LOG3classification_report](https://user-images.githubusercontent.com/28285099/205476590-b8fe610e-4e80-4882-9196-8d028b1deca1.png)

All in all, looks solid with an extremely high f1 score 0.99.

# What to Run

analysis_logistic_regression_3.py is what to run for the model, but for every other attempt, feel free to check out the SVM and KNN models.
I ultimately decided to stick with simple, to get the most straightforward results.

# Disclaimer

I, being new to data science, did not personally know how to program all of this from scratch. Hence, I have used the internet to my advantage so that I was able to provide these visualizations. Here are the websites I have used to help me on this project.

https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
https://towardsdatascience.com/support-vector-machines-explained-with-python-examples-cb65e8172c85

Thank you.





