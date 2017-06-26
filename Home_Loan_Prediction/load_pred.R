library(plyr)
library(xlsx)
library(ggplot2)
library(Metrics)
library(caret)
library(randomForest)
library(foreach)
#Xgboost
library(xgboost)
library(readr)
library(stringr)
library(caret)


train <- read.csv("train.csv")
train$source <- c("train")

test <- read.csv("test.csv")
test$Loan_Status <- NA
test$source <- c("test")

data <- rbind(train,test)

columns.fact <- c("Gender","Education","Married","Dependents","Self_Employed","Credit_History","Property_Area","Loan_Status")
columns.train <- c("Loan_ID","Gender","Education","Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area","Loan_Status","source")
columns.test <- c("Gender","Education","Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area","source")

data1<-data[,columns.train]
write.csv(data1, "loan_pred.csv", row.names = FALSE)

# Missing values
table(is.na(data1))
colSums(is.na(data1))
#md.pattern(data)

# Unique values
apply(data1, 2, function(x)length(unique(x)))

#Analizing frequency in categories in each nominal variable
apply(subset(data1, select = columns.fact), 2, function(x) {count(x)})

# -------------imputing Gender ----------
Idx_NA_Gender <- which(data1$Gender == "")
data1$Gender[Idx_NA_Gender] <- c("Male")

levels(data1$Gender)[levels(data1$Gender)== c("Male")] <- c(1)  
levels(data1$Gender)[levels(data1$Gender)== c("Female")] <- c(0) 

# ----------- Education -------------------
levels(data1$Education)[levels(data1$Education) == c("Graduate")] <- c(1)  
levels(data1$Education)[levels(data1$Education) == c("Not Graduate")] <- c(0) 

# -------------imputing Married ----------
Idx_NA_Married <- which(data1$Married == "")
data1$Married[Idx_NA_Married] <- c("Yes")
levels(data1$Married)[levels(data1$Married) == c("Yes")] <- c(1)  
levels(data1$Married)[levels(data1$Married) == c("No")] <- c(0) 

# -------------imputing Dependents ---------
Idx_NA_Dep1 <- which(data1$Dependents == "")
Idx_NA_Dep1 <- which(is.na(data1$Dependents))
data1$Dependents[Idx_NA_Dep1] <- c(0)
levels(data1$Dependents)[levels(data1$Dependents)== c("3+")] <- c("3")

# -------------imputing Self Employed ---------improve this
Idx_NA_Self <- which(data1$Self_Employed == "")
data1$Self_Employed[Idx_NA_Self] <- c("No")
levels(data1$Self_Employed)[levels(data1$Self_Employed) == c("Yes")] <- c(1)  
levels(data1$Self_Employed)[levels(data1$Self_Employed) == c("No")] <- c(0) 

# Applicant income, outlier treatment -----------

ApplicantIncome <- data1$ApplicantIncome
qnt <- quantile(ApplicantIncome, probs=c(.25, .75), na.rm = T)
caps <- quantile(ApplicantIncome, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(ApplicantIncome, na.rm = T)
data1$ApplicantIncome[data1$ApplicantIncome < (qnt[1] - H)] <- mean(ApplicantIncome)
data1$ApplicantIncome[data1$ApplicantIncome > (qnt[2] + H)] <- mean(ApplicantIncome)

#data1$ApplicantIncome[data1$ApplicantIncome < (qnt[1] - H)] <- caps[1]
#data1$ApplicantIncome[data1$ApplicantIncome > (qnt[2] + H)] <- caps[2]

# ---  CoapplicantIncome, replacing values with zero   -------------------------

df.Coap <- data.frame(gender = data1$Gender,  coapin = data1$CoapplicantIncome)
Mean_Coap <- ddply(df.Coap, "gender", summarise, mean = mean(coapin, na.rm = TRUE))
data1$CoapplicantIncome[data1$Gender==1 & data1$CoapplicantIncome == 0] <- Mean_Coap$mean[2]
data1$CoapplicantIncome[data1$Gender==0 & data1$CoapplicantIncome == 0] <- Mean_Coap$mean[1]

# ------------- LoanAmount -------------------+
df.LoanA <- data.frame(gender = data1$Gender,  loan = data1$LoanAmount)
Mean_Loan <- ddply(df.LoanA, "gender", summarise, mean = mean(loan, na.rm = TRUE))
data1$LoanAmount[data1$Gender==1 & is.na(data1$LoanAmount)] <- Mean_Loan$mean[2]
data1$LoanAmount[data1$Gender==0 & is.na(data1$LoanAmount)] <- Mean_Loan$mean[1]

# -- Loan_Amount_Term ----

df.LoanAT <- data.frame(education = data1$Education,  loan = data1$LoanAmount)
Mean_LoanAT <- ddply(df.LoanAT, "education", summarise, mean = mean(loan, na.rm = TRUE))

Idx_NA_LAT <- which(is.na(data1$Loan_Amount_Term))
data1$Loan_Amount_Term[data1$Education==1 & is.na(data1$Loan_Amount_Term)] <- Mean_LoanAT$mean[1]
data1$Loan_Amount_Term[data1$Education==0 & is.na(data1$Loan_Amount_Term)] <- Mean_LoanAT$mean[2]

# -- Credit History ----

#Idx_NA_CH <- which(is.na(data1$Credit_History))
data1$Credit_History[is.na(data1$Credit_History) & data1$Education == 1] <- c(1)
data1$Credit_History[is.na(data1$Credit_History) & data1$Education == 0] <- c(0)

# --- replace Property_Area and Dependents --- 

# Dummy -- One Hot Encoding
columns.dum <- c("Property_Area", "Dependents")
data1 <- dummy.data.frame(data1, names = columns.dum, sep='_')

# Loan status

levels(data1$Loan_Status)[levels(data1$Loan_Status) == c("Y")] <- c(1)
levels(data1$Loan_Status)[levels(data1$Loan_Status) == c("N")] <- c(0)


# Adding new feature, LAIN resume ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term
#data1$LAIN <- (data1$LoanAmount/(data1$ApplicantIncome+data1$CoapplicantIncome)/2)*data1$Loan_Amount_Term

# Normalize 
data1$TotalIncome <- (data1$ApplicantIncome + data1$CoapplicantIncome)/2
data1$TotalIncome <- Normalize(data1$TotalIncome)
data1$ApplicantIncome <- Normalize(data1$ApplicantIncome)
data1$CoapplicantIncome <- Normalize(data1$CoapplicantIncome)
data1$LoanAmount <- Normalize(data1$LoanAmount)
data1$Loan_Amount_Term <- Normalize(data1$Loan_Amount_Term)


train <- subset(data1, data1$source == "train")

columns.train <- !names(data1) %in% c("Loan_ID","source","ApplicantIncome","CoapplicantIncome")
train.final <- train[,columns.train]
train.final$Loan_Status <- as.factor(train.final$Loan_Status)
rownames(train.final) <- NULL  

test <- subset(data1, data1$source == "test")

columns.test <- !names(data1) %in% c("Loan_ID","source","Loan_Status", "ApplicantIncome","CoapplicantIncome")
test.final <- test[, columns.test]
rownames(test.final) <- NULL  

write.csv(train.final, "train_modified.csv", row.names = FALSE)
write.csv(test.final, "test_modified.csv",  row.names = FALSE)

# ------------------------Random Forest----------------------------------

set.seed(1000)
#seeds <- vector(mode = "list", length = 50)
#for(i in 1:150) seeds[[i]] <- sample.int(300000, 11)
mycontrol <- trainControl(method = "cv", number = 2)
mygrid <- expand.grid(mtry = c(5:16)) 

rforest_model <- train(Loan_Status ~ ., data = train.final, method="rf", metric="Accuracy", maximize = TRUE, tuneGrid=mygrid, trControl=mycontrol, ntree = 350, nodesize = 77,importance = TRUE, replace = TRUE, do.trace = TRUE, norm.votes=FALSE)
print(rforest_model)


pred <- predict(rforest_model,test.final)
pred <- as.factor(pred)

levels(pred)[levels(pred) == c("1")] <- c("Y")
levels(pred)[levels(pred) == c("0")] <- c("N")
  
base <- data.frame(Loan_ID = as.factor(test$Loan_ID),  Loan_Status = pred)
write.table(base, "solLoan_Forestmodel.csv", sep = ",", row.names = FALSE, quote = FALSE) 


#---------------- ANN--------------

control <- trainControl(method = "cv", number = 3)
#number of hidden neurons http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
my.grid <- expand.grid(.decay = c(0.5:0.1), .size = c(2:10))

nnet_model <- train(Loan_Status ~ ., data = train.final, method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = T, metric="Accuracy", maximize = TRUE)  
print(nnet_model)

pred <- predict(nnet_model,test.final)
pred <- as.factor(pred)

levels(pred)[levels(pred) == c("1")] <- c("Y")
levels(pred)[levels(pred) == c("0")] <- c("N")

base <- data.frame(Loan_ID = as.factor(test$Loan_ID),  Loan_Status = pred)
write.table(base, "solLoan_nnet.csv", sep = ",", row.names = FALSE, quote = FALSE) 


#---------------- Xgboost ----------------------------------

xgb <- xgboost(data = train.final, 
               label = y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)



# ------------- Functions ----------------

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)] # en el dataset no aparecen con NA sino con ""
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

WriteResults <- function(file, pred)
{ 
  base <- test[,c("Loan_ID")]
  base$Loan_Status <- pred
  rownames(base) <- NULL  
  write.table(base, file, sep = ",", row.names = FALSE, quote = FALSE)  
}

Normalize <- function(vector)
{
  min <- min(vector)
  max <- max(vector)
  vector <- (vector-min)/(max-min)
  vector
  
}


