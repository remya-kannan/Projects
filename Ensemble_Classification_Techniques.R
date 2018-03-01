############################Case Study 1#####################################
############################Setting path#####################################
setwd("/Users/remya/Desktop/Depaul_4th_quarter/CSC_529_Advanced_Data_Mining/Project/Data")
############################Reading the data#################################
hd_c <- read.table("hd_cleveland.txt", header = FALSE, sep = ',')
hd_num <- read.table("hd_cleveland.txt", header = FALSE, sep = ',')
############################Exploratory data analysis########################
nrow(hd_c)
summary(hd_c)
colnames(hd_c) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num")
colnames(hd_num) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num")
hd_c$num <- as.factor(ifelse(hd_c$num==0,"No","Yes"))
hd_num$V12 <-as.numeric(hd_num$V12)
hd_num$V13 <-as.numeric(hd_num$V13)
library(devtools)
install_github("ujjwalkarn/xda")
library(xda)
#univariate analysis
numSummary(hd_num)
Plot(hd_num, 'num')
boxplot(hd_num$num, main="Boxplot for outlier detection", col="blue")
#bivariate analysis 
library(corrplot)
m <- cor(hd_num)
corrplot(m, method = "circle")
#Multivariate analysis
#PCA on variables to see which account for greatest variance 
hd_pca<- prcomp(hd_num, center = TRUE, scale. = TRUE) 
print(hd_pca)
plot(hd_pca, type="l", main="PCA plot on the importance of the attributes")
summary(hd_pca)
##########################Libraries########################################
library(mlbench)
library(caret)
library(caretEnsemble)
##########################Boosting Algorithms##############################
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 1234
metric <- "Accuracy"
#C5.0
set.seed(seed)
m_c50 <- train(num~., data=hd_c, method="C5.0", metric=metric, trControl=control)
m_c50
#Gradient Boosting
set.seed(seed)
m_gbm <- train(num~., data=hd_c, method="gbm", metric=metric, trControl=control, verbose=FALSE)
m_gbm
#summarize results
boosting_results <- resamples(list(c5.0=m_c50, gbm=m_gbm))
summary(boosting_results)
dotplot(boosting_results, main="Summary of results: c5.0 vs Gradient Boosting")
##########################Bagging algorithms##################################
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 1234
metric <- "Accuracy"
#Bagged CART
set.seed(seed)
m_treebag <- train(num~., data=hd_c, method="treebag", metric=metric, trControl=control)
m_treebag
#Random Forest
library(randomForest)
set.seed(seed)
m_rf <- train(num~., data=hd_c, method="rf", metric=metric, trControl=control)
m_rf
#summarize results
bagging_results <- resamples(list(treebag=m_treebag, rf=m_rf))
summary(bagging_results)
dotplot(bagging_results, main="Summary of the results: Bagged CART vs Random Forest")
##########################Stacking algorithms##################################
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
library(MASS)
model <- caretList(num~., data=hd_c, trControl=control, methodList=algorithmList, metric=metric)
result <- resamples(model)
summary(result)
dotplot(result)
#correlation between models
modelCor(result)
splom(result)
#stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(model, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
#stack using random forest
set.seed(seed)
stack.rf <- caretStack(model, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)
##########################Random forest####################################
#run ntree
set.seed(1234)
#creating an index upon which to split data into train and test sets
ind <- sample(2, nrow(hd_c), replace=TRUE, prob=c(0.8, 0.2))
train <- hd_c[ind==1,]
nrow(train)
test <- hd_c[ind==2,]
nrow(test)
library(rpart)
#########################Building a singular model tree##################### 
tree <- rpart(num ~ . , data=train)
pred.tree <- predict(tree, test, type = "class")
table(pred.tree,test$num)
t <- table(pred.tree,test$num)
#examing the missclassifcation cases
confusionMatrix(t)
plot(tree)
text(tree)
#######################Building a model using random forest###################
samplehd_c <- sample(nrow(train), replace = T)
u <- unique(samplehd_c)
#confirming the sampling process was completed
length(u)
length(samplehd_c)
#ntree=30
randF <- randomForest(num ~., data = train, ntree = 30)
table(predict(randF),train$num)
print(randF)
plot(randF)
#ntree=50
randF <- randomForest(num ~., data = train, ntree = 50)
table(predict(randF),train$num)
print(randF)
plot(randF)
#ntree=75--ideal depth
m_rF <- randomForest(num ~., data = train, ntree = 75)
table(predict(m_rF),train$num)
print(m_rF)
plot(m_rF)
#selecting optimal number of variables at each split
mtry <- tuneRF(train[-14], train$num, ntreeTry = 75, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = TRUE)
best.m <- mtry[mtry[,2]==min(mtry[,2]),1]
print(mtry)
print(best.m)
#Run random forest on the optimal number of variables selected
randF <- randomForest(num ~., data = train, mtry=best.m, importance=TRUE, ntree = 75)
table(predict(randF),train$num)
print(randF)
plot(randF, main="Random forest plot with optimal attributes")
#Build random forest for test data
randpred <- predict(randF, newdata=test)
table(randpred, test$num)
postResample(randpred, test$num)
#Calculate predictive probabilities of training dataset.
pred=predict(randF,type = "prob")
pred=predict(randF,type = "prob",test)
# Plot partial dependence of each predictor
par(mfrow = c(3, 5), mar = c(2, 2, 2, 2), pty = "s");
for (i in 1:(ncol(train) - 1))
{
  partialPlot(randF, train, names(train)[i], xlab = names(train)[i],
              main = NULL);
}
###############################Model Evaluation##################################
#Calculate variable importance
importance(randF)
varImpPlot(randF, main="Plot of the variable importance of the random forest", col="red")
#Evaluate the performance of the random forest for classification.
pred2=predict(randF,type = "prob")
library(ROCR)
#prediction is ROCR function
perf = prediction(pred2[,2], train$num)
#AUC
auc = performance(perf, "auc")
auc
#True Positive and Negative Rate
pred3 = performance(perf, "tpr","fpr")

#ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

#Prediction on the test set
pred.randF <- predict(randF, test, type = "class")
tf <- table(pred.randF,test$num)
confusionMatrix(tf)






