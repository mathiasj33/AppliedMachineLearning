library('klaR')
dat <- read.csv(file="../data/pima-indians-diabetes.csv", header=FALSE)
labels <- dat$V9
dat$V9 <- NULL

numTest <- round(0.2 * nrow(dat), 0)
avgAcc <- 0
for(i in 1:10) {
  testIndices <- sample(1:nrow(dat), numTest, replace=FALSE)
  test <- dat[testIndices,]
  train <- dat[-testIndices,]
  test_labels <- labels[testIndices]
  train_labels <- labels[-testIndices]
  svm <- svmlight(train, train_labels)
  y <- predict(svm, test)[['class']]
  acc <- 0
  for(i in 1:length(test_labels)) {
    if(y[[i]] == test_labels[[i]]) acc <- acc + 1
  }
  acc <- acc/length(test_labels)
  avgAcc <- avgAcc + acc
}
avgAcc <- avgAcc/10