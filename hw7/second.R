library(glmnet)
data = read.csv("data/genes.csv", header=F, sep="")
data = t(data)
x = as.matrix(data)

labels = read.csv("data/genes_labels.csv", header=F, sep="")
labels$V1 = lapply(labels$V1, function(x) if(x > 0) 0 else 1)
y = data.matrix(labels)

model = cv.glmnet(x, y, family = "binomial", type.measure = "class")
plot(model)
p = predict(model, x, type = "class", s = model$lambda.min)
p = apply(p, 2, as.numeric)

most_common_sum = if(sum(y == 0) > sum(y == 1)) sum(y == 0) else sum(y == 1)
baseline = most_common_sum / length(y)
accuracy = sum (p == y) / length(p)
