library(caTools)
library(caret)
library(e1071)
library(gpairs)
library(ggplot2)
library(ModelMetrics)
library(ROCR)

#  Mean Function
mean <- function(data){
  sum <- 0
  for(i in 1:length(data)){
    sum <- sum + data[i]
  }
  return(sum/length(data))
}

#  Standard Deviation
std <- function(data,mean){
  sum <- 0
  for(i in 1:length(data)){
    sum <- sum + (data[i]-mean)^2
  }
  return(sum/length(data))
}

# Normalization
norm = function(x, meanVal, stdVal){
  return (exp(-(x-meanVal)^2/(2*stdVal^2))/(stdVal*sqrt(2*pi)))
}

# ----------------------------> Naive Bayes Classifier <-----------------------------------------------
predict <- function(train_data,test_x,category_cols,continous_cols){
  
  category_lkh <- list()
  
  for(i in category_cols){
    category_lkh[[i]] <- list()
    for(j in as.vector(unique(train_data[i]))[[i]]){
      s = as.name(as.character(j))
      category_lkh[[i]][[s]] <- c(length(which(train_data[i]==j & train_data$target==0))/length(which(train_data$target==0)),length(which(train_data[i]==j & train_data$target==1))/length(which(train_data$target==1)))
    }
    # <- temp
  }
  #category_lkh
  
  train_data.y0 <- subset(train_data,subset=train_data$target==0)
  train_data.y1 <- subset(train_data,subset=train_data$target==1)
  continous_lkh <- list()
  for(i in continous_cols){
    continous_lkh[[i]] <- list()
    mean <- mean(train_data.y0[[i]])
    std <- std(train_data.y0[[i]],mean)
    continous_lkh[[i]][["0"]] <- list("mean" = mean,"std" = std)
    mean <- mean(train_data.y1[[i]])
    std <- std(train_data.y1[[i]],mean)
    continous_lkh[[i]][["1"]] <- list("mean" = mean,"std" = std)                                 
  }
  
  pred_y = c()
  prior_target = c(length(which(train_data$target==0))/nrow(train_data),length(which(train_data$target==1))/nrow(train_data))
  for(row in 1:nrow(test_x)){
    prob.0 <- prior_target[1] # Posterior Probability P(target=0/x)
    prob.1 <- prior_target[2]
    
    target <- 0
    for(i in category_cols){
      test_val <- as.character(test_x[row,i])
      prob.0 <- prob.0 *  category_lkh[[i]][[test_val]][1]
      prob.1 <- prob.1 *  category_lkh[[i]][[test_val]][2]
    }
    for(i in continous_cols){
      test_val <- test_x[row,i]
      prob.0 <- prob.0 * norm(test_val,continous_lkh[[i]][["0"]][["mean"]],continous_lkh[[i]][["0"]][["std"]])
      prob.1 <- prob.1 * norm(test_val,continous_lkh[[i]][["1"]][["mean"]],continous_lkh[[i]][["1"]][["std"]])
    }
    if(prob.1>prob.0){
      target<-1
    }else{
      target<-0
    }
    pred_y[row] <- target
  }
  return(pred_y)
}



# ---------------------------> R-fold cross validation <-------------------------

KFoldCrossVal <- function(dataset,k) {
  sprintf("Cross Validation with K = %d",k)
  len = nrow(dataset)
  accuracy = c()
  for(i in 1:k){
    from <- 1+as.integer(len/k)*(i-1)
    to <- as.integer(len/k)*i
    test_data <- dataset[from:to,]
    train_data <- rbind(dataset[0:(from-1),],dataset[(to+1):len,])
    
    test_x <- subset(test_data,select = -target)
    #print(test_x)
    test_y <- subset(test_data,select = target)
    
    result <- data.frame(test_y)
    result$predicted_target <- predict(train_data,test_x,category_cols,continous_cols)
    #confusion_matrix <- table(result$target,result$predicted_target)
    #print(confusion_matrix)
    accuracy[i] <- length(which(result$target==result$predicted_target))/length(result$target)*100
  }
  return(mean(accuracy))
}


# -----------------------> Importing Dataset <----------------------------------- 
dataset = read.csv("heart.csv")
dataset = na.omit(dataset)
str(dataset)
summary(dataset)

continous_cols = c("age","trestbps","chol","thalach","oldpeak")
category_cols = c("sex","cp","fbs","restecg","exang","slope","ca","thal")

gpairs(subset(dataset[,1:14],select=c(continous_cols,"target")))


#--------> Scatter Plots of Continuous Attributes <---------------------
ScatterPlot <- function(x,y){
  ggplot(dataset, aes(.data[[x]], .data[[y]])) + geom_point(aes(color = target)) + 
    scale_x_continuous(x)+
    scale_y_continuous(y)+
    theme_bw() + labs(title=paste("Scatterplot",x,"V/S",y))
}
ScatterPlot("age","trestbps")
ScatterPlot("age","chol")
ScatterPlot("age","thalach")
ScatterPlot("age","oldpeak")
ScatterPlot("trestbps","chol")
ScatterPlot("trestbps","thalach")
ScatterPlot("trestbps","oldpeak")
ScatterPlot("chol","thalach")
ScatterPlot("chol","oldpeak")
ScatterPlot("thalach","oldpeak")


# ----------> K Fold Cross Validation <----------
avgAcc = c()
least.k = 3
max.k= 15
for(k in least.k:max.k){
  accuracy <- KFoldCrossVal(dataset,k)
  avgAcc<- append(avgAcc,accuracy)
}

print(avgAcc)
bestKvalue = which.max(avgAcc)+least.k-1
print(bestKvalue)

sprintf("So the acuracy with best k = %d value for k fold cross validation is %f",bestKvalue,avgAcc[[bestKvalue-least.k+1]])




#----------------------------Predicting with total data-----------------------------------

split <- createDataPartition(dataset$target,p=0.75,list=FALSE)
td.size <- 0.75*nrow(dataset)
train_data <- dataset[1:td.size,]
test_data <- dataset[td.size:nrow(dataset),]
test_x <- subset(test_data, select = -target)
test_y <- subset(test_data, select = target)

#contingency table from an training dataset comprising the prior and posterior probabilities.
for(i in category_cols){
  prior.ct <- prop.table(table(train_data$target,train_data[[i]],dnn = list("target",i)))
  cat("Prior Probabilties of target and",i,"\n")
  print(prior.ct)
  cat("\n")
  post.ct <- prop.table(table(train_data$target,train_data[[i]],dnn = list("target",i)),margin = 2)
  cat("Posterior Probabilties of target and",i,"\n")
  print(post.ct)
  cat("\n")
}

result <- data.frame(test_y)
result$predicted_target <- predict(train_data,test_x,category_cols,continous_cols)
write.csv(result,"result.csv", row.names = FALSE)

# Accuracy 
accuracy <- length(which(result$target==result$predicted_target))/length(result$target)*100
sprintf("accuracy: %f",accuracy)

# RMSE Error
error <- rmse(result$target,result$predicted_target)
sprintf("root mean square error: %f",error)

# Confusion Matrix
confusion_matrix <- table(result$target,result$predicted_target,dnn=names(result))
print(confusion_matrix)

# Precision
precision <- ppv(result$target,result$predicted_target)
sprintf("Precision: %f",precision)
# Recall
recall <- recall(result$target,result$predicted_target)
sprintf("recall: %f",recall)

# F1 Score 
f1.score <- f1Score(result$target,result$predicted_target)
sprintf("F1 score: %f",f1.score)

pred <- prediction(result$target,result$predicted_target)

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec")

plot (RP.perf)

# ROC curve
ROC.perf <- performance(pred, "tpr", "fpr")
plot (ROC.perf)

# ROC area under the curve
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
sprintf('Area Under Curve(AUC) value : %f',auc)
