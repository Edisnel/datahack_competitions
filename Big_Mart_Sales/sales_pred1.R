library(plyr)
library(xlsx)
library(ggplot2)
library(Metrics)
library(dplyr)

train <- read.csv("train_modified.csv")
test <- read.csv("test_modified.csv")

mean_sales <- mean(train$Item_Outlet_Sales)

base = test[,c("Item_Identifier","Outlet_Identifier")]
base$Item_Outlet_Sales <- round(mean_sales,2)
rownames(base) <- NULL

columns.train <- c("Item_Weight","Item_Fat_Content","Item_Visibility","Item_MRP","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Years","Item_Outlet_Sales")
columns.test <- c("Item_Weight","Item_Fat_Content","Item_Visibility","Item_MRP","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Years")

final_train <- subset(train, select = columns.train)
final_test <- subset(test, select = columns.test)
rownames(final_train) <- NULL
rownames(final_test) <- NULL

# --------------------- Linear regression -----------------------
linear_model <- lm(Item_Outlet_Sales ~ ., data = final_train)
summary(linear_model)

par(mfrow=c(2,2))
plot(linear_model)

rmse(final_train$Item_Outlet_Sales, linear_model$fitted.values)

# Prediction with test data...
pred <- predict(linear_model, test)
WriteResults("sol_linreg.csv", pred)

#------------------- Random Forest ---------------------------------

library(randomForest)
library(foreach)

library(dummies)

columns.dum <- c('Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type_New','Item_Fat_Content')
numeric_train <- dummy.data.frame(final_train, names = columns.dum, sep='_')
numeric_test <- dummy.data.frame(final_test, names = columns.dum, sep='_')

columns.train <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Years','Item_Outlet_Sales')
columns.test <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Years')

numeric_train <- numeric_train[,columns.train]
numeric_test <- numeric_test[,columns.test]

colnames(numeric_train) <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier1','Outlet_Type_SupermarketType1','Outlet_Years','Item_Outlet_Sales')
colnames(numeric_test) <-  c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier1','Outlet_Type_SupermarketType1','Outlet_Years')

seeds <- vector(mode = "list", length = 50)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)

#cartGrid <- expand.grid(.nodesize=(90:150), mtry="5")


mycontrol <- trainControl(method = "cv", number = 5, seeds = seeds)
rforest_model <- train(Item_Outlet_Sales ~ ., data = numeric_train, method = "parRF", trControl = mycontrol, ntree = 500, nodesize = 100, importance = TRUE, replace = TRUE, metric = "RMSE", maximize = FALSE, corr.bias = TRUE)
print(rforest_model)

#rforest_model <- randomForest(Item_Outlet_Sales ~ ., data = numeric_train, control = mycontrol, mtry = 4, ntree = 500, importance = TRUE, replace = TRUE, nodesize = 100, maxnodes = 100, corr.bias = TRUE)
#print(rforest_model)

pred <- predict(rforest_model)
rmse(numeric_train$Item_Outlet_Sales, pred)

#varImpPlot(forest_model)

pred <- predict(rforest_model,numeric_test)
#numeric_train$prediction <- pred

WriteResults("sol_Forestmodel.csv", pred)


#-----------------eXtreme Gradient Boosing (xgboost) ----------------



#-----------------Artificial Neural Network(ANN)-------------------

library(dummies)
library(nnet)

columns.dum <- c('Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type_New','Item_Fat_Content')
numeric_train <- dummy.data.frame(final_train, names = columns.dum, sep='_')
numeric_test <- dummy.data.frame(final_test, names = columns.dum, sep='_')

columns.train <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Years','Item_Outlet_Sales')
columns.test <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Years')

numeric_train <- numeric_train[,columns.train]
numeric_test <- numeric_test[,columns.test]

control <- trainControl(method = "cv", number = 5)
#number of hidden neurons http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5,5))

normalvalue <- max(numeric_train$Item_Outlet_Sales) - min(numeric_train$Item_Outlet_Sales)

#linout is the activation function
nnet_model <- train(Item_Outlet_Sales ~ ., data = numeric_train, method = "nnet", maxit = 500, tuneGrid = my.grid, trace = T)  
print(nnet_model)

nnet.predict <- predict(nnet_model,numeric_test)*normalvalue

rmse(numeric_train$Item_Outlet_Sales, nnet.predict)

#write.csv(numeric_train, "train_numeric.csv",  row.names = FALSE)
#write.csv(numeric_test, "test_numeric.csv",  row.names = FALSE)

nnetResult <- read.csv("knimeResult.csv") # read from file generated by Knime
pred <- nnetResult$new.column
WriteResults("sol_nnetmodel.csv", pred)


#---------------- Functions -------------------

WriteResults <- function(file, pred)
{ 
  base = test[,c("Item_Identifier","Outlet_Identifier")]
  base$Item_Outlet_Sales <- round(pred,4)
  rownames(base) <- NULL  
  write.table(base, file, sep = ",", row.names = FALSE, quote = FALSE)  
}






  







