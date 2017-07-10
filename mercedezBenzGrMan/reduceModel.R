# File: reduceModel.R
# Auth: uhkniazi
# Date: 05/07/2017
# Desc: reduce the model size by selecting important variables


library(LearnBayes)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

set.seed(123) # for replication

dfData = read.csv('mercedezBenzGrMan/DataExternal/train.csv', header=T)
dim(dfData)
head(dfData)
dfData = dfData[,-1]

## keep 30% of the data as validation set
i = sample(1:nrow(dfData), size = nrow(dfData)*0.30, replace = F)
dfData.val = dfData[i,]
dfData.train = dfData[-i,]

library(randomForest)
set.seed(123)


fit.rf = randomForest(y ~ ., data=dfData.train)
# get variables importance
varImpPlot(fit.rf)
dfRF = data.frame(importance(fit.rf))
head(dfRF)
ivScore = dfRF$IncNodePurity
names(ivScore) = rownames(dfRF)
ivScore = sort(ivScore, decreasing = T)
head(ivScore)
hist(ivScore)
# take the top 20 variables
length(ivScore)
ivScore = ivScore[1:20]
tail(ivScore)

## take a subset of this data
dfData.train.rf1 = dfData.train[,c('y', names(ivScore))]

dfData = dfData.train.rf1
dim(dfData)
## take a subset of the training data to fit model
i = sample(1:nrow(dfData), size = nrow(dfData)*0.30, replace = F)
dfData.pr = dfData[-i,]
dfData = dfData[i,]
## make model matrix and fit model
m = model.matrix(y ~ ., data=dfData)
dim(m)
stanDso = rstan::stan_model(file='mercedezBenzGrMan/studentTMixtureRegression.stan')

lStanData = list(Ntotal=nrow(dfData), y=dfData$y, iMixtures=2, Ncol=ncol(m), X=m)

plot(density(dfData.train$y))
plot(density(dfData$y))
plot(density(dfData.val$y))
## give initial values if you want, look at the density plot 
initf = function(chain_id = 1) {
  list(mu = c(90, 120), sigma = c(11, 11*2), iMixWeights=c(0.5, 0.5), nu=c(3, 3))
} 

fit.stan = sampling(stanDso, data=lStanData, iter=1200, chains=1, init=initf, cores=1, pars=c('sigma', 'iMixWeights', 'betasMix1',
                                                                                             'betasMix2', 'mu', 'nu'))
print(fit.stan, digi=3)
#traceplot(fit.stanTMix)

## get fitted values
m = extract(fit.stan)
names(m)

## get the coefficients
iModel.1 = c(mean(m$mu[,1]), apply(m$betasMix1, 2, mean))
iModel.2 = c(mean(m$mu[,2]), apply(m$betasMix2, 2, mean))
iMixWeights = apply(m$iMixWeights, 2, mean)

## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.train = mean((iAggregate - dfData$y)^2)

## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.test.pr = mean((iAggregate - dfData.pr$y)^2)

## try on the validation set
## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.val = mean((iAggregate - dfData.val$y)^2)

#### refit the model again but on a different and larger subset of the data to see if error rate improves
dfData = dfData.train.rf1
dim(dfData)
## take a subset of the training data to fit model
i = sample(1:nrow(dfData), size = nrow(dfData)*0.80, replace = F)
dfData.pr = dfData[-i,]
dfData = dfData[i,]
## make model matrix and fit model
m = model.matrix(y ~ ., data=dfData)
dim(m)

lStanData = list(Ntotal=nrow(dfData), y=dfData$y, iMixtures=2, Ncol=ncol(m), X=m)

## give initial values if you want, look at the density plot 
initf = function(chain_id = 1) {
  list(mu = c(90, 120), sigma = c(11, 11*2), iMixWeights=c(0.5, 0.5), nu=c(3, 3), betasMix1=iModel.1[-1],
       betasMix2=iModel.2[-1])
} 

fit.stan2 = sampling(stanDso, data=lStanData, iter=1200, chains=1, init=initf, cores=1, pars=c('sigma', 'iMixWeights', 'betasMix1',
                                                                                              'betasMix2', 'mu', 'nu'))
print(fit.stan2, digi=3)
#traceplot(fit.stanTMix)

## get fitted values
m = extract(fit.stan2)
names(m)

## get the coefficients
iModel.1 = c(mean(m$mu[,1]), apply(m$betasMix1, 2, mean))
iModel.2 = c(mean(m$mu[,2]), apply(m$betasMix2, 2, mean))
iMixWeights = apply(m$iMixWeights, 2, mean)

## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.train2 = mean((iAggregate - dfData$y)^2)

## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.test.pr2 = mean((iAggregate - dfData.pr$y)^2)

## try on the validation set
## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.2
## get aggregate
iAggregate = cbind(iPred1, iPred2)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.val2 = mean((iAggregate - dfData.val$y)^2)

###########################################################
#### 3 component distribution
###########################################################
dfData = dfData.train.rf1
dim(dfData)
## take a subset of the training data to fit model
i = sample(1:nrow(dfData), size = nrow(dfData)*0.80, replace = F)
dfData.pr = dfData[-i,]
dfData = dfData[i,]
## make model matrix and fit model
m = model.matrix(y ~ ., data=dfData)
dim(m)
stanDso = rstan::stan_model(file='mercedezBenzGrMan/studentTMixtureRegression3Components.stan')

lStanData = list(Ntotal=nrow(dfData), y=dfData$y, iMixtures=3, Ncol=ncol(m), X=m)

## give initial values if you want, look at the density plot 
initf = function(chain_id = 1) {
  list(mu = c(75, 90, 120), sigma = c(5, 11, 11*2), iMixWeights=c(0.3, 0.3, 0.4), nu=c(3, 3, 3))
} 

fit.stan3 = sampling(stanDso, data=lStanData, iter=2000, chains=1, init=initf, cores=1, pars=c('sigma', 'iMixWeights', 'betasMix1',
                                                                                              'betasMix2', 'betasMix3', 'mu', 'nu',
                                                                                              'betaSigma'))
print(fit.stan3, digi=3)
#traceplot(fit.stanTMix)

## get fitted values
m = extract(fit.stan3)
names(m)

## get the coefficients
iModel.1 = c(mean(m$mu[,1]), apply(m$betasMix1, 2, mean))
iModel.2 = c(mean(m$mu[,2]), apply(m$betasMix2, 2, mean))
iModel.3 = c(mean(m$mu[,3]), apply(m$betasMix3, 2, mean))

iMixWeights = apply(m$iMixWeights, 2, mean)

## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData) %*% iModel.2
iPred3 = model.matrix(y ~ ., data=dfData) %*% iModel.3

## get aggregate
iAggregate = cbind(iPred1, iPred2, iPred3)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.train3 = mean((iAggregate - dfData$y)^2)

## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.2
iPred3 = model.matrix(y ~ ., data=dfData.pr) %*% iModel.3
## get aggregate
iAggregate = cbind(iPred1, iPred2, iPred3)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.test.pr3 = mean((iAggregate - dfData.pr$y)^2)

## try on the validation set
## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.2
iPred3 = model.matrix(y ~ ., data=dfData.val[,colnames(dfData.pr)]) %*% iModel.3
## get aggregate
iAggregate = cbind(iPred1, iPred2, iPred3)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.val3 = mean((iAggregate - dfData.val$y)^2)




