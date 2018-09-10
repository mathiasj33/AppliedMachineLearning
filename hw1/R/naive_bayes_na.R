calcAcc <- function(train, test) {
  pos_prob <- sum(train$V9) / length(train$V9)
  neg_prob <- 1-pos_prob
  pmeans <- lapply(subset(train, V9 == 1), mean, na.rm=TRUE)
  pvars <- lapply(subset(train, V9 == 1), function(x) mean((x-mean(x, na.rm=TRUE))^2, na.rm=TRUE))
  nmeans <- lapply(subset(train, V9 == 0), mean, na.rm=TRUE)
  nvars <- lapply(subset(train, V9 == 0), function(x) mean((x-mean(x, na.rm=TRUE))^2, na.rm=TRUE))
  
  calcPos <- function(dp) {
    sum <- 0
    for(i in 1:8) {
      if(is.na(dp[[i]])) next
      sum <- sum + log(dnorm(dp[[i]], pmeans[[i]], pvars[[i]]))
    }
    sum <- sum + log(pos_prob)
  }
  
  calcNeg <- function(dp) {
    sum <- 0
    for(i in 1:8) {
      if(is.na(dp[[i]])) next
      sum <- sum + log(dnorm(dp[[i]],nmeans[[i]], nvars[[i]]))
    }
    sum <- sum + log(neg_prob)
  }
  
  pred <- function(dp) {
    return(if(calcPos(dp) >= calcNeg(dp)) 1 else 0)
  }
  
  preds <- apply(test, 1, pred)
  acc <- 0
  for(i in 1:nrow(test)) {
    if(preds[i] == test$V9[i]) acc <- acc + 1
  }
  acc <- acc/nrow(test)
  return(acc)
}

dat <- read.csv(file="../data/pima-indians-diabetes.csv", header=FALSE)
dat$V3[dat$V3 == 0] <- NA
dat$V4[dat$V4 == 0] <- NA
dat$V6[dat$V6 == 0] <- NA
dat$V8[dat$V8 == 0] <- NA

numTest <- round(0.2 * nrow(dat), 0)
avgAcc <- 0
for(i in 1:10) {
  testIndices <- sample(1:nrow(dat), numTest, replace=FALSE)
  test <- dat[testIndices,]
  train <- dat[-testIndices,]
  avgAcc <- avgAcc + calcAcc(train, test)
}
avgAcc <- avgAcc/10
