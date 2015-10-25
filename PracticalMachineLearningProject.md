Practical Machine Learning - Course Project
========================================================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

The Class of action can be one of the following: exactly according to the specification (A), 
throwing elbows to the front (B), lifting the dumbbell only halfway (C), lowering the dumbbell only halfway (D), throwing the hips to the front (E).

## Getting Data


```r
downloadFiles<-function(
    dataURL="", destF="t.csv"
    ){
        if(!file.exists(destF)){
            download.file(dataURL, destF, method="curl")
        }else{
            message("data already downloaded.")
        }
    }
```


```r
trainURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
downloadFiles(trainURL, "pml-training.csv")
```

```
## data already downloaded.
```

```r
downloadFiles(testURL, "pml-testing.csv")
```

```
## data already downloaded.
```

```r
training <- read.csv("pml-training.csv",na.strings=c("NA",""))
testing <-read.csv("pml-testing.csv",na.strings=c("NA",""))
```


```r
dim(training)
```

```
## [1] 19622   160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

As we see, there are 19622 observations and 160 variables

## Preprocessing

We would do cross validation in order to ensure a good predicting model. In our training subset we will split our training data in a subset for create the model and one subset to validate the results before doing it with the test subset. 


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.1
```

```r
set.seed(333)
trainset <- createDataPartition(training$classe, p = 0.7, list = FALSE)
training <- training[trainset, ]
Validation <- training[-trainset, ]
```

### Removing zero variance variables


```r
nzvcol <- nearZeroVar(training)
training <- training[, -nzvcol]
```


### Removing columns that has a lot of missing values and columns that are descriptive (these won't help us to create a model)


```r
cntlength <- sapply(training, function(x) {
    sum(!(is.na(x) | x == ""))
})
nullcol <- names(cntlength[cntlength < 0.6 * length(training$classe)])
descriptcol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window")
excludecols <- c(descriptcol, nullcol)
training <- training[, !names(training) %in% excludecols]
```

## Model with Random Forest


```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rfModel <- randomForest(classe ~ ., data = training, importance = TRUE, ntrees = 10)
```

### Validation


```r
ptraining <- predict(rfModel, training)
print(confusionMatrix(ptraining, training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

### Cross - Validation


```r
pvalidation <- predict(rfModel, Validation)
print(confusionMatrix(pvalidation, Validation$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1180    0    0    0    0
##          B    0  799    0    0    0
##          C    0    0  700    0    0
##          D    0    0    0  668    0
##          E    0    0    0    0  768
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2868     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2868   0.1942   0.1701   0.1623   0.1866
## Detection Rate         0.2868   0.1942   0.1701   0.1623   0.1866
## Detection Prevalence   0.2868   0.1942   0.1701   0.1623   0.1866
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

## Test set prediction for the Submission section of the Project


```r
ptest <- predict(rfModel, testing)
ptest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


```r
answers <- as.vector(ptest)

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```

## References

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.
