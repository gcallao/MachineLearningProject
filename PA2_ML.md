#Practical Machine Learning: Course Project
By gcallao

##1. Synopsis
Using the data from a Human Activity Recognition project published in http://groupware.les.inf.puc-rio.br/har we developed a prediction model using Random Forest as learning method for classification and Cross Validation for feature selection. Our final model makes use of 28 features and 500 trees, obtaining 0.38% as out of sample error rate. Finally we predict the class of the 20 observations in the testing data set for submission in the course project webpage achieving a 100% level of accuracy.

##2. Data Processing

We first load the data downloaded using the links provided in the course project webpage.

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

And give a quick look using the first 02 observations of the training data set.

```r
head(training,2)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         11      1.41       8.07    -94.4                3
## 2         no         11      1.41       8.07    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                                                         
## 2                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                                                                      NA
## 2                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1             NA                         NA             NA             
## 2             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                  NA                   NA                   
## 2                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                   NA            NA               NA            NA
## 2                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1             NA                NA             NA           NA
## 2             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1              NA           NA         0.00            0        -0.02
## 2              NA           NA         0.02            0        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1          -21            4           22            -3           599
## 2          -22            4           22            -7           608
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1          -313     -128      22.5    -161              34            NA
## 2          -311     -128      22.5    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 1           NA              NA           NA            NA               NA
## 2           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 1            NA          NA             NA          NA        0.00
## 2            NA          NA             NA          NA        0.02
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 1        0.00       -0.02        -288         109        -123         -368
## 2       -0.02       -0.02        -290         110        -125         -369
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 1          337          516                                     
## 2          337          513                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 1                                                                       
## 2                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 1           NA            NA          NA           NA            NA
## 2           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 1          NA                 NA                  NA                NA
## 2          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1         13.05         -70.49       -84.87                       
## 2         13.13         -70.64       -84.71                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 1                                                                     
## 2                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 1                                                              NA
## 2                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 1                 NA                                 NA                 NA
## 2                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 1                                       NA                       NA
## 2                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                                          37                 NA
## 2                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                NA                   NA                NA
## 2                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 1                 NA                    NA                 NA
## 2                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 1               NA                  NA               NA                0
## 2               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 1            -0.02                0             -234               47
## 2            -0.02                0             -233               47
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 1             -271              -559               293               -65
## 2             -269              -555               296               -64
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 1         28.4         -63.9        -153                      
## 2         28.3         -63.9        -153                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 1                                                                  
## 2                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 1                                                           NA
## 2                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 1                NA                               NA                NA
## 2                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 1                                     NA                      NA
## 2                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 1                                        36                NA
## 2                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1               NA                  NA               NA                NA
## 2               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                   NA                NA              NA
## 2                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                 NA              NA            0.03               0
## 2                 NA              NA            0.02               0
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.02             192             203            -215
## 2           -0.02             192             203            -216
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1              -17              654              476      A
## 2              -18              661              473      A
```

We discard the first 05 variables because they are related to the id of the observation, the user name, and time data as saw previously. They have no use for our purposes.

```r
training <- training[,-c(1:5)]
```

Also we observed that there are missing values in numerous variables as NAs or empty values, so we proceed with some feature cleaning. In this case we discard the features with a frequency above 95% of being empty values or NAs. We record the indices of those features in delVarIndex and its names in delColNames.

```r
delColNames <- c()
delVarIndex <- c()
tempvalues <- c()

for (i in c(1:dim(training)[2])){
        tempframe <- data.frame(table(training[,i], useNA = "always")/dim(training)[1])
        tempframe$Var1 <- as.character(tempframe$Var1)
        
        if (tempframe[1,1] == ""){
            tempframe[1,1] <- "NoValue"
        }
        
        tempframe$Var1[is.na(tempframe$Var1)] <- "NoValue"              
        tempvalue <- sum(subset(tempframe, Var1 == "NoValue")[,2])
        
        if (tempvalue >= 0.95){
                delColNames <- append(delColNames, colnames(training)[i])
                delVarIndex <- append(delVarIndex, i)
                tempvalues <- append(tempvalues, tempvalue)
        }
}
```

And we now apply the Index vector to the training data, keeping only 54 features to train the model in this case (the 55th correspond to the label variable, classe).

```r
training <- training[, -delVarIndex]
dim(training)
```

```
## [1] 19622    55
```

As final validation, we decide to partition the training data in 02 subsets, training1 and testing1, almost equally sized with p=0.5. In this testing01 data set we are going to estimate the out of sample error rate using the model fit that we are going to develop in the training1 data set.

```r
library(lattice); library(ggplot2); library(caret); library(randomForest)

set.seed(1)
InTrain<-createDataPartition(y=training$classe,p=0.5,list=FALSE)

training1 <-training[InTrain,]
testing1 <- training[-InTrain,]
```

##3. Prediction Model

We select **Random Forest** as learning method for classification using **10-fold Cross Validation** as training control method, and print the modelFit and the finalModel result.

```r
modelFit <- train(classe ~ ., data = training1, method = "rf", trControl = trainControl(method = "cv"))
modelFit
```

```
## Random Forest 
## 
## 9812 samples
##   54 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 8832, 8831, 8832, 8829, 8831, 8831, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9910    0.9887  0.004503     0.005698
##   28    0.9957    0.9946  0.002346     0.002968
##   54    0.9930    0.9911  0.002426     0.003068
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

```r
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.39%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2787    2    0    0    1    0.001075
## B    8 1888    2    1    0    0.005793
## C    0    8 1703    0    0    0.004676
## D    0    0    9 1598    1    0.006219
## E    0    0    0    6 1798    0.003326
```

We see in the modelFit output that because its level of accuracy the optimal model for prediction consist in 28 features selected of the total 54 available initialy for training, and 500 trees with an out of bag error rate of 0.39%.

Now finally we use the testing1 data set (a subset of the training data) to estimate the **out of sample error rate**, obtaining **0.38%**.

```r
table(testing1$classe, predict(modelFit, testing1[,-55]))
```

```
##    
##        A    B    C    D    E
##   A 2788    2    0    0    0
##   B    3 1890    3    2    0
##   C    0    3 1708    0    0
##   D    0    0   13 1595    0
##   E    0    2    0    9 1792
```

```r
paste("Out of Sample Error Rate =", as.character(round((1-sum(testing1$classe == predict(modelFit, testing1[,-55]))/length(testing1$classe))*100,2)), "%")
```

```
## [1] "Out of Sample Error Rate = 0.38 %"
```

##4. Prediction on the Testing Data

As confident as we are because the low level of out of sample error, we use the final model developed to classify the testing data for submission in the course project webpage.

But first we clean the testing data as it was done in the training data set.

```r
testing <- testing[,-c(1:5)]
testing <- testing[,-delVarIndex]
```

Now we apply the model fit to the testing data.

```r
Prediction <- predict(modelFit, testing[,-55])
Prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

We define the function provided in the course project webpage to generate the text files to submit.

```r
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
```

And generate those files.

```r
pml_write_files(Prediction)
```

#5. Conclussion
We use Random Forest as learning method for classification and 10-fold cross validation as training control method to develop a prediction model using the Human Activity Recognition Data. Our out of sample error was 0.38% and finally we estimate the class of the observations in the testing data achieving an accuracy level of 100%.







