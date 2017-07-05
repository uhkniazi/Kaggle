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

dfData = dfData.train
## take a subset of the training data to fit model
i = sample(1:nrow(dfData), size = nrow(dfData)*0.30, replace = F)
dfData = dfData[i,]

## make model matrix and fit model
m = model.matrix(y ~ ., data=dfData)

stanDso = rstan::stan_model(file='mercedezBenzGrMan/studentTMixtureRegression.stan')

lStanData = list(Ntotal=nrow(dfData), y=dfData$y, iMixtures=2, Ncol=ncol(m), X=m)

plot(density(dfData.train$y))
plot(density(dfData$y))
plot(density(dfData.val$y))
## give initial values if you want, look at the density plot 
initf = function(chain_id = 1) {
  list(mu = c(90, 120), sigma = c(11, 11*2), iMixWeights=c(0.5, 0.5), nu=c(3, 3))
} 

fit.stan = sampling(stanDso, data=lStanData, iter=600, chains=2, init=initf, cores=1)
print(fit.stanTMix, digi=3)
traceplot(fit.stanTMix)

## check if labelling degeneracy has occured
## see here: http://mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html
params1 = as.data.frame(extract(fit.stan, permuted=FALSE)[,1,])
params2 = as.data.frame(extract(fit.stan, permuted=FALSE)[,2,])
params3 = as.data.frame(extract(fit.stan, permuted=FALSE)[,3,])
params4 = as.data.frame(extract(fit.stan, permuted=FALSE)[,4,])

## check if the means from different chains overlap
## Labeling Degeneracy by Enforcing an Ordering
par(mfrow=c(2,2))
plot(params1$`mu[1]`, params1$`mu[2]`, pch=20, col=2)
plot(params2$`mu[1]`, params2$`mu[2]`, pch=20, col=3)
plot(params3$`mu[1]`, params3$`mu[2]`, pch=20, col=4)
plot(params4$`mu[1]`, params4$`mu[2]`, pch=20, col=5)

par(mfrow=c(1,1))
plot(params1$`mu[1]`, params1$`mu[2]`, pch=20, col=2, xlim=c(3, 28), ylim=c(28, 42))
points(params2$`mu[1]`, params2$`mu[2]`, pch=20, col=3)
points(params3$`mu[1]`, params3$`mu[2]`, pch=20, col=4)
points(params4$`mu[1]`, params4$`mu[2]`, pch=20, col=5)