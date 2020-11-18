Background
==========

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>
(see the section on the Weight Lifting Exercise Dataset).

The Data
========

The training data for this project are available here:
<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a>

The test data are available here:
<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv</a>

The data for this project come from this source:
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>

Data Preparation
================

    # Load required libraries
    library(caret)
    library(corrplot)
    library(dplyr)
    library(e1071)
    library(ggplot2)
    library(kernlab)
    library(knitr)
    library(lattice)
    library(plyr)
    library(randomForest)
    library(rattle)
    library(rpart)
    library(rpart.plot)
    library(RColorBrewer)

    # Read data files
    train_df <- read.csv('pml-training.csv', na.strings = c('NA', '#DIV/0!', ''))
    test_df <- read.csv('pml-testing.csv', na.strings = c('NA', '#DIV/0!', ''))

    dim(train_df)

    ## [1] 19622   160

    dim(test_df)

    ## [1]  20 160

We can see that there are 19,622 observations in the training dataset,
vs. 20 observations in the testing dataset. There are 160 variables in
both dataset.

Data Cleaning
=============

Let us remove rows with missing data, and drop variables we are not
interested in.

    train_df <- train_df[, colSums(is.na(train_df)) == 0]
    test_df <- test_df[, colSums(is.na(test_df)) == 0]
    train_df <- train_df[, -c(1:7)]
    test_df <- test_df[, -c(1:7)]

Model & Validation
==================

We split the cleaned training dataset training-validation set with a
ratio of 70%-30%, so as to compute out-of-sample errors/ cross
validation. We will be applying 2 different models: (1) Random Forest,
(2) SVM. Seed will be used for reproducability purposes.

    set.seed(128)
    inTrain <- createDataPartition(train_df$classe, p = .7, list = F)
    train <- train_df[inTrain, ]
    valid <- train_df[-inTrain, ]

Random Forest
-------------

Let’s train and test our model using Random Forest.

    train$classe <- factor(train$classe)
    model_RF <- randomForest(classe ~ ., data = train, method = 'class')
    valid_RF <- predict(model_RF, valid, type = 'class')
    confusionMatrix(valid_RF, as.factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    3    0    0    0
    ##          B    0 1134    3    0    0
    ##          C    0    2 1020    7    0
    ##          D    0    0    3  957    2
    ##          E    0    0    0    0 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9966          
    ##                  95% CI : (0.9948, 0.9979)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9957          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9956   0.9942   0.9927   0.9982
    ## Specificity            0.9993   0.9994   0.9981   0.9990   1.0000
    ## Pos Pred Value         0.9982   0.9974   0.9913   0.9948   1.0000
    ## Neg Pred Value         1.0000   0.9989   0.9988   0.9986   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1927   0.1733   0.1626   0.1835
    ## Detection Prevalence   0.2850   0.1932   0.1749   0.1635   0.1835
    ## Balanced Accuracy      0.9996   0.9975   0.9961   0.9959   0.9991

We can see that estimated accuracy for Random Forest is 99.7% and
estimated out-of-sample error is ~0.3%.

SVM
---

Let’s now train and test our model using SVM.

    model_SVM <- svm(classe ~ ., data = train)
    valid_SVM <- predict(model_SVM, valid)
    confusionMatrix(valid_SVM, as.factor(valid$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1667   83    3    0    0
    ##          B    1 1011   29    0    8
    ##          C    6   42  979   78   27
    ##          D    0    1   10  886   24
    ##          E    0    2    5    0 1023
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9458          
    ##                  95% CI : (0.9397, 0.9514)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9313          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9958   0.8876   0.9542   0.9191   0.9455
    ## Specificity            0.9796   0.9920   0.9685   0.9929   0.9985
    ## Pos Pred Value         0.9509   0.9638   0.8648   0.9620   0.9932
    ## Neg Pred Value         0.9983   0.9735   0.9901   0.9843   0.9878
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2833   0.1718   0.1664   0.1506   0.1738
    ## Detection Prevalence   0.2979   0.1782   0.1924   0.1565   0.1750
    ## Balanced Accuracy      0.9877   0.9398   0.9614   0.9560   0.9720

We can see that estimated accuracy for Random Forest is only 94.6% and
estimated out-of-sample error is ~5.4%.

Conclusion
==========

Given the accuracy results above, we can see that Random Forest is a
better predicting model than SVM.

Prediction
----------

As such, below is the prediction results for 20 different test cases
using Random Forest.

    predict_RF <- predict(model_RF, test_df, type = 'class')
    predict_RF

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
