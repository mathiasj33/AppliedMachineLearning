data = read.csv("data/mice.csv")

gender_data = data[,c(2, 4:41)]
gender_data = gender_data[complete.cases(gender_data),]
gender_labels = gender_data$sex
gender_data = gender_data[,-c(1)]

x = as.matrix(gender_data)
y = as.matrix(gender_labels)

model = cv.glmnet(x, y, family = "binomial", type.measure = "class")
plot(model)
p = predict(model, x, type = "class", s = model$lambda.min)
accuracy = sum(p == y) / length(p)

most_common_sum = if(sum(y == "f") > sum(y == "m")) sum(y == "f") else sum(y == "m")
baseline = most_common_sum / length(y)
