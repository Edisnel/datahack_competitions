library(plyr)
library(xlsx)
library(ggplot2)
library(Metrics)
library(dplyr)

library(rpart)
library(e1071)
library(rpart.plot)
library(caret)

#imputing missing values #library(mice) #library(mi)
#library(Hmisc) 
#library(mice)

#sales prediction

train <- read.csv("Train.csv")
train$source <- c("train")

test <- read.csv("Test.csv")
test$Item_Outlet_Sales <- NA
test$source <- c("test")

data <- rbind(train,test)

data1<-data

# Missing values
table(is.na(data1))
colSums(is.na(data1))
#md.pattern(data)

# Unique values
apply(data1, 2, function(x)length(unique(x)))

#Analizing frequency in categories in each nominal variable
apply(subset(data1, select = c("Item_Identifier","Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type")), 2, function(x) {count(x)})

# -------------impute Item_Weight by the average weight of the particular item----------
df.Weight <- data.frame(Item_Identifier = data1$Item_Identifier,  Item_Weight = data1$Item_Weight)
Mean_Weight <- ddply(df.Weight, "Item_Identifier", summarise, mean_weight = round(mean(Item_Weight, na.rm = TRUE),4))

Idx_NA_Weight <- which(is.na(data1$Item_Weight))
Identifiers <- data1$Item_Identifier[Idx_NA_Weight]
data1$Item_Weight[Idx_NA_Weight] <- Mean_Weight$mean_weight[match(Identifiers,Mean_Weight$Item_Identifier)]

# -------------impute Outlet_Size by the mode of the particular Outlet_Type--------------
df.OuletSize <- data.frame(Outlet_Type = data1$Outlet_Type,  Outlet_Size = data1$Outlet_Size)
df.Outlet_Mode <- ddply(df.OuletSize, "Outlet_Type", summarise, outlet_mode = Mode(Outlet_Size,na.rm = TRUE))

Idx_NA_OutLetSize <- which(data1$Outlet_Size == "")
df.NA_OuletSize <- df.OuletSize[which(df.OuletSize$Outlet_Size == ""),]

# fill empty rows in Outlet_Size with the mode.
data1$Outlet_Size[Idx_NA_OutLetSize] <- sapply(df.NA_OuletSize$Outlet_Type, function(x) ModeByOutletType(x))

# --------minimum value here is 0, it has no sense...impute it with mean visibility of that product---
summary(data1$Item_Visibility)

df.visibility <- data.frame(Item_Identifier = data1$Item_Identifier,  Item_Visibility= data1$Item_Visibility)
#Mean visibility by Item_Identifier
Mean_Visibility <- ddply(df.visibility, "Item_Identifier", summarise, mean_visibility = round(mean(Item_Visibility),4))

#Replacing each null value of Item_Visibility by mean value corresponding to Item_Identifier
Idx_zero_visibility <- which(data1$Item_Visibility == 0)
Identifiers <- data1$Item_Identifier[Idx_zero_visibility]

data1$Item_Visibility[Idx_zero_visibility] <- sapply(Identifiers,function(x) fMean_Visibility(x))


# -----------New column with the years of operation of a store ----------
data1$Outlet_Years = 2013 - data1$Outlet_Establishment_Year

# --------------Modify categories of Item_Fat_Content--------------
levels(data1$Item_Fat_Content)[levels(data1$Item_Fat_Content)=="low fat"] <- "Low Fat"
levels(data1$Item_Fat_Content)[levels(data1$Item_Fat_Content)=="LF"] <- "Low Fat"
levels(data1$Item_Fat_Content)[levels(data1$Item_Fat_Content)=="reg"] <- "Regular"

# ------------------------New Item Type ---------------------------------

q <- substr(data1$Item_Identifier,1,2)
q <- gsub("FD","Food",q)
q <- gsub("DR","Drinks",q)
q <- gsub("NC","Non-Consumable",q)
table(q)
data1$Item_Type_New <- q # remember this variable for model building


# ---------------- See OUTLIERS----------------------



# ---------- Label Encoding and One Hot Encoding ----------- pending



# ----------------------- Exporting data -------------------------------------

train <- subset(data1, data1$source == "train")
test <- subset(data1, data1$source == "test")

columns.train <- c("Item_Weight","Item_Fat_Content","Item_Visibility","Item_MRP","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Years","Item_Outlet_Sales")
columns.test <- c("Item_Weight","Item_Fat_Content","Item_Visibility","Item_MRP","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Years")

final_train <- subset(train, select = columns.train)
final_test <- subset(test, select = columns.test)
rownames(final_train) <- NULL
rownames(final_test) <- NULL

write.csv(train, "train_modified.csv", row.names = FALSE)
write.csv(test, "test_modified.csv",  row.names = FALSE)


# ----------------- Some graphics for data exploration-visualization
g <- ggplot(train, aes(x= Item_Visibility, y = Item_Outlet_Sales)) 
g <- g + geom_point(size = 2.5, color="navy") + xlab("Item Visibility")
g <- g + ylab("Item Outlet Sales") + ggtitle("Item Visibility vs Item Outlet Sales")
g

g <- ggplot(train, aes(x= Item_Weight, y = Item_Outlet_Sales)) 
g <- g + geom_point(size = 2.5, color="navy") + xlab("Item Weight")
g <- g + ylab("Item Outlet Sales") + ggtitle("Item Weight vs Item Outlet Sales")
g

g <- ggplot(train, aes(x= Outlet_Years, y = Item_Outlet_Sales)) 
g <- g + geom_point(size = 2.5, color="navy") + xlab("Outlet Years")
g <- g + ylab("Item Outlet Sales") + ggtitle("Outlet_Years vs Item Outlet Sales")
g

g <- ggplot(train, aes(Item_Type, Item_Outlet_Sales)) 
g <- g + geom_bar( stat = "identity") +theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "navy")) 
g <- g + xlab("Item Type") + ylab("Item Outlet Sales")+ggtitle("Item Type vs Sales")
g

#---------------------------MODEL BUILDING-------------------------------
#------------------- Random Forest ---------------------------------

library(randomForest)
library(foreach)

library(dummies)

columns.dum <- c('Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type_New','Item_Fat_Content')
numeric_train <- dummy.data.frame(final_train, names = columns.dum, sep='_')
numeric_test <- dummy.data.frame(final_test, names = columns.dum, sep='_')

columns.train <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2','Outlet_Years','Item_Outlet_Sales')
columns.test <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier 1','Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2','Outlet_Years')

numeric_train <- numeric_train[,columns.train]
numeric_test <- numeric_test[,columns.test]

colnames(numeric_train) <- c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier1','Outlet_Type_SupermarketType1','Outlet_Type_SupermarketType2','Outlet_Years','Item_Outlet_Sales')
colnames(numeric_test) <-  c('Item_Weight', 'Item_Visibility','Item_MRP','Outlet_Size_High','Outlet_Size_Medium','Outlet_Location_Type_Tier1','Outlet_Type_SupermarketType1','Outlet_Type_SupermarketType2','Outlet_Years')

seeds <- vector(mode = "list", length = 50)
for(i in 1:150) seeds[[i]] <- sample.int(10000, 500)

mycontrol <- trainControl(method = "cv", number = 5, seeds = seeds)
mygrid <- expand.grid(mtry = 5) 

rforest_model <- train(Item_Outlet_Sales ~ ., data = numeric_train, method = "parRF", tuneGrid = mygrid, trControl = mycontrol, ntree = 500, nodesize = 100, maxnodes = 100, importance = TRUE, metric = "RMSE", maximize = FALSE, replace = TRUE, do.trace = TRUE)
print(rforest_model)

pred <- predict(rforest_model)
rmse(numeric_train$Item_Outlet_Sales, pred)

pred <- predict(rforest_model,numeric_test)

WriteResults("sol_Forestmodel.csv", pred)


# ------------- Functions ----------------


Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[x != ""] # en el dataset no aparecen con NA sino con ""
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

ModeByOutletType <- function(OutletType)
{
  mode <- df.Outlet_Mode$outlet_mode[which(df.Outlet_Mode$Outlet_Type == OutletType)]
  mode
}

fMean_Visibility <- function(Id)
{
  m <- Mean_Visibility[which(Mean_Visibility$Item_Identifier == Id),]$mean_visibility
  m
}

WriteResults <- function(file, pred)
{ 
  base = test[,c("Item_Identifier","Outlet_Identifier")]
  base$Item_Outlet_Sales <- round(pred,4)
  rownames(base) <- NULL  
  write.table(base, file, sep = ",", row.names = FALSE, quote = FALSE)  
}

