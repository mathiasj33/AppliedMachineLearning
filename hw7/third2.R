data = read.csv("data/mice.csv")

data = data[,c(1, 4:41)]
data = data[complete.cases(data),]
for(label in data$strain) {
  count = sum(data$strain == label)
  if(count < 10) {
    data = data[data$strain != label,]
  }
}
labels = droplevels(data$strain)
data = data[,-c(1)]

x = as.matrix(data)
y = as.matrix(labels)

model = cv.glmnet(x, y, family = "multinomial", type.measure = "class")
plot(model)
p = predict(model, x, type = "class", s = model$lambda.min)
accuracy = sum(p == y) / length(p)
baseline = 1 / length(labels)
