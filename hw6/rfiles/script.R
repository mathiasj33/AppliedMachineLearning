library("MASS")
do_plot = T
data = read.csv("../data/housing.data", header=F, sep="")

train_model = function(data) {
  model = lm(V14 ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13, data=data)
  return(model)
}

plot_stdres = function(model, transform_back=F, lambda=NULL) {
  stdres = rstandard(model)
  fitted = if(transform_back) from_boxcox(fitted(model), lambda) else fitted(model)
  plot(fitted, stdres, xlab="Fitted value", ylab="Normalized residual")
}

plot_hist = function(model) {
  stdres = rstandard(model)
  hist(stdres, breaks=20)
}

to_boxcox = function(y, lambda) {
  if(lambda == 0) {
    return(log(y))
  } else {
    return((y ^ lambda - 1)/lambda)
  }
}

from_boxcox = function(y, lambda) {
  if(lambda == 0) {
    return(exp(y))
  } else {
    return((lambda * y + 1)^(1/lambda))
  }
}

get_best_lambda = function(bc) {
  return(bc$x[which.max(bc$y)])
}

untransformed_model = train_model(data)

data = data[-c(369,373,372,370,371,366,368,365,413),]
model = train_model(data)

bc = boxcox(model, plotit=do_plot)

lambda = get_best_lambda(bc)
bcdata = data.frame(data)
bcdata$V14 = NULL
bcdata$V14 = to_boxcox(data$V14, lambda)
bcmodel = train_model(bcdata)

if(do_plot) {
  plot(untransformed_model)
}