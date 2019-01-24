library(glmnet)
data = read.csv("data/blogData_train.csv", header = F)
# data = data[sample(nrow(data), 10),]
x = as.matrix(data[,1:280])
y = as.matrix(data[,281])

#model = cv.glmnet(x, y, family = "poisson")
#save(model, file = "first_model.rda")
model = load("first_model.rda")
plot(model)
p = floor(predict(model, x, type = "response", s = model$lambda.min))
plot(y,p,xlab = "true", ylab = "predicted")

test_data = NULL
files = list.files(path = "data/blog_test", pattern="*.csv")
for(f in files) {
  path = paste("data/blog_test/", f, sep = "")
  if(is.null(test_data)) {
    test_data = read.csv(path, header = F)
  } else {
    test_data = rbind(test_data, read.csv(path, header = F))
  }
}
x = as.matrix(test_data[,1:280])
y = as.matrix(test_data[,281])
p = floor(predict(model, x, type = "response", s = model$lambda.min))
plot(y,p,xlab = "true", ylab = "predicted")
