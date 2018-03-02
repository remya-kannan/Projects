############################Case Study 2#####################################
############################Setting path#####################################
setwd("../Case_study_2")
############################Reading the data#################################
train <- read.table("trial_1.csv", sep=",", header = TRUE, stringsAsFactors = FALSE, na.strings=c("NA", ""))
test <- read.table("trial_2.csv", sep=",", header = TRUE, stringsAsFactors = FALSE, na.strings=c("NA", ""))
############################Libraries########################################
library('ggplot2')
library('scales')
library('ggthemes')
library(dplyr)
library(kernlab)
library(e1071)
library(ROCR)
############################Exploring the data###############################
nrow(train)
nrow(test)
summary(train)
rm(list=ls())
set.seed(12345)
test$Survived <- NA
full <- rbind(train, test)
dim(train); dim(test); dim(full)
colnames(train)
str(train)
###########################Univariate Analysis###############################
#Pclass
ggplot(train, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(position = 'dodge') +
  labs(x = 'Pclass') +
  ggtitle("Figure 1", subtitle = "Survival of Passenger by Class")

#Sex
full$gender <- as.factor(full$Sex)
full$gender <- relevel(full$gender, ref="female")

#Cabin
full$cabin_deck <- toupper(substring(full$Cabin, 1, 1))
table(full$Survived, full$cabin_deck, useNA="ifany")
full[full$cabin_deck %in% c('A', 'G', 'T'), ]$cabin_deck <- 'AGT'
full[full$cabin_deck %in% c('B', 'D', 'E'), ]$cabin_deck <- 'BDE'
full[full$cabin_deck %in% c('C', 'F'), ]$cabin_deck <- 'CF'
full[is.na(full$Cabin), ]$cabin_deck <- "unknown"
full$cabin_deck <- as.factor(full$cabin_deck)

#Title
title <- unique(gsub("^.+, (.+?)\\. .+$", "\\1", full$Name))
title
temp1 <- c("Dona", "Jonkheer", "the Countess", "Sir", "Lady", "Don")
temp2 <- c("Col", "Capt", "Major", "Dr")
full$title <- gsub("^.+, (.+?)\\. .+$", "\\1", full$Name)
full[full$title == "Mlle", ]$title <- "Miss"
full[full$title == "Mme" | full$title == "Ms", ]$title <- "Mrs"
full[full$title %in% temp1, ]$title <- "noble"
full[full$title %in% temp2, ]$title <- "pros"
full$title <- as.factor(full$title)

ggplot(full[1:891,], aes(x = title, fill = factor(Survived))) +
  geom_bar(position = 'dodge') +
  labs(x = 'Title') +
  ggtitle("Figure 2", subtitle = "Survival based on Title")


#Name
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

#Fare
med_fare <- full %>% group_by(Pclass) %>% summarize(medians=median(Fare, na.rm=T))

#Ticket
full$ticket_str <- gsub("(\\D*)\\d+", "\\1", full$Ticket)
full[full$ticket_str == "", ]$ticket_str <- "unavailable"

ggplot(full[1:891,], aes(x = ticket_num, fill = factor(Survived))) +
  geom_bar(position = 'dodge') +
  labs(x = 'Ticket number') +
  ggtitle("Figure 3", subtitle = "Survival of passengers based on thier ticket number")


#Embarked
full$embarked <- as.factor(full$Embarked)
full[is.na(full$embarked), ]$embarked <- 'S'

#Passenger ID
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

full$Embarked[c(62, 830)] <- 'C'

#Age 
full$age <- full$Age
medians <- full %>% group_by(title) %>% summarize(medians=median(Age, na.rm=T))
full <- inner_join(full, medians, by='title')
full[is.na(full$age), ]$age <- full[is.na(full$age), ]$medians

##########################Bivariate Analysis###############################
#SibSp, Parch
full$family <- full$SibSp + full$Parch + 1

ggplot(full[1:891,], aes(x = family, fill = factor(Survived))) +
  geom_bar(position = 'dodge') +
  labs(x = 'Fsize') +
  ggtitle("Figure 4", subtitle ="Survival based on the size of the family")

full$Fsize <- full$SibSp + full$Parch + 1
full$Family <- paste(full$Surname, full$Fsize, sep='_')

ggplot(full[1:891,], aes(x = Family, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

#SUrvival of family
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  facet_grid(.~Sex) + 
  theme_few()

#Child and mother
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'
table(full$Child, full$Survived)
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'
table(full$Mother, full$Survived)


full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

#########################Linear kernel SVM##############################

# Use tune() to do 10-fold CV
model_linear <- tune(svm, Survived ~ Pclass + Sex + Age + Child + Sex * Pclass + SibSp + Parch + Family + Mother, data = train, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(model_linear)

bestmod <- model_linear$best_model
summary(bestmod)

model_predict_lenar <- predict(bestmod, test)

###########################RBF kernel SVM###############################
#First iteration: default tuning parameters
model_radial_1 <- train(as.factor(Survived)~.Pclass + Sex + Age + Child + Sex * Pclass + SibSp + Parch + Family + Mother, data=train, method="svmRadial")
model_radial_1
plot(model_radial_1)

#Second iteration: manually tune parameters 
radial_grid = expand.grid(C=c(0.5,0.75,0.9,1,1.25), sigma=c(0.01,0.02,0.03,0.04,0.05,0.1,0.12,0.15,0.2))
set.seed(123456) 
model_radial_2 <- train(as.factor(Survived)~.Pclass + Sex + Age + Child + Sex * Pclass + SibSp + Parch + Family + Mother, data=train, method="svmRadial", tuneGrid=radial_grid)
model_radial_2
plot(model_radial_2)

#Third iteration: optimum parameters for robustness
radial_grid = expand.grid(C=0.75,sigma=0.15)
set.seed(123456) 
model_radial_3 <- train(as.factor(Survived)~.Pclass + Sex + Age + Child + Sex * Pclass + SibSp + Parch + Family + Mother, data=train,method="svmRadial", tuneGrid=radial_grid)
model_radial_3
model_radial_pred = predict(model_radial_3, newdata=test, type="raw")
confusionMatrix(model_radial_pred, test$Survived)

###########################Model Evaluation###############################
#prediction is ROCR function
perf = prediction(model_radial_pred[,2], test$Survived)

#AUC
auc = performance(perf, "auc")
auc

pred = performance(perf, "tpr","fpr")

#ROC curve
plot(pred,col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2)

