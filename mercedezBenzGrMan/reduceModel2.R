# File: reduceModel2.R
# Auth: uhkniazi
# Date: 05/07/2017
# Desc: reduce the model size by selecting important variables


library(LearnBayes)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

##################### try with CCrossvalidation library
if(!require(downloader) || !require(methods)) stop('Library downloader and methods required')

url = 'https://raw.githubusercontent.com/uhkniazi/CCrossValidation/experimental/CCrossValidation.R'
download(url, 'CCrossValidation.R')

# load the required packages
source('CCrossValidation.R')
# delete the file after source
unlink('CCrossValidation.R')



dfDataMainTest = read.csv('mercedezBenzGrMan/DataExternal/test.csv', header=T)
dfData = read.csv('mercedezBenzGrMan/DataExternal/train.csv', header=T)
## some factors are missing levels between training and test sets
dim(dfData)
head(dfData)
dfData = dfData[,-1]

iResponse = dfData$y
dfData = dfData[,-1]

## need to reduce the number of variables for RF or it takes too long
## lets drop the correlated variables
## find correlated variables
m = NULL;

for (i in 1:ncol(dfData)){
  m = cbind(m, dfData[,i])
}
colnames(m) = colnames(dfData)
v = apply(m, 2, var)
f = which(v <= 0)
length(f)
# drop variables with low variance
mCor = cor(m[,-f], use="pairwise.complete.obs")
library(caret)
### find the columns that are correlated and should be removed
n = findCorrelation((mCor), cutoff = 0.7, names=T)
data.frame(n)
## drop these variables
m = colnames(m[,-f])
m = m[!(m %in% n)]
cvTopVariables.cor = m

# perform a random forest now
dfData = dfData[,cvTopVariables.cor]
dim(dfData)
## create validation set and training set
## keep 30% of the data as validation set
set.seed(123) # for replication
iTest = sample(1:nrow(dfData), size = nrow(dfData)*0.30, replace = F)
plot(density(iResponse[iTest]))
plot(density(iResponse[-iTest]))

oCVran = CVariableSelection.RandomForest(dfData[-iTest,], groups = iResponse[-iTest], 
                                         boot.num = 100, big.warn = F)
save(oCVran, file='mercedezBenzGrMan/Results/oCVran.rds')

## choose the top variables and see if they 
## have similar levels in test set
df = CVariableSelection.RandomForest.getVariables(oCVran)

cvTopVariables.ran = rownames(df)[1:30]

str(dfData[,cvTopVariables.ran])
str(dfDataMainTest[,cvTopVariables.ran])

## use rstan now to fit the model
dfData = dfData[,cvTopVariables.ran]
dfData$y = iResponse

## make model matrix and fit model
m = model.matrix(y ~ ., data=dfData[-iTest,])
dim(m)


###########################################################
#### 3 component distribution
###########################################################
stanDso = rstan::stan_model(file='mercedezBenzGrMan/studentTMixtureRegression3Components.stan')

lStanData = list(Ntotal=nrow(dfData[-iTest,]), y=dfData$y[-iTest], iMixtures=3, Ncol=ncol(m), X=m)

## give initial values if you want, look at the density plot 
initf = function(chain_id = 1) {
  list(mu = c(75, 90, 120), sigma = c(5, 11, 11*2), iMixWeights=c(0.3, 0.3, 0.4), nu=c(3, 3, 3))
} 

fit.stan3 = sampling(stanDso, data=lStanData, iter=2000, chains=1, init=initf, cores=1, pars=c('sigma', 'iMixWeights', 'betasMix1',
                                                                                              'betasMix2', 'betasMix3', 'mu', 'nu',
                                                                                              'betaSigma'))
print(fit.stan3, digi=3)

save(fit.stan3, file='mercedezBenzGrMan/Results/fit.stan3.rds')
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
iPred1 = model.matrix(y ~ ., data=dfData[-iTest,]) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData[-iTest,]) %*% iModel.2
iPred3 = model.matrix(y ~ ., data=dfData[-iTest,]) %*% iModel.3

## get aggregate
iAggregate = cbind(iPred1, iPred2, iPred3)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.train3 = mean((iAggregate - dfData$y[-iTest])^2)
plot(iAggregate, dfData$y[-iTest], pch=20)
summary(lm(iAggregate ~ dfData$y[-iTest]))

## repeat on the test and validation data
## calculate training error
iPred1 = model.matrix(y ~ ., data=dfData[iTest,]) %*% iModel.1
iPred2 = model.matrix(y ~ ., data=dfData[iTest,]) %*% iModel.2
iPred3 = model.matrix(y ~ ., data=dfData[iTest,]) %*% iModel.3
## get aggregate
iAggregate = cbind(iPred1, iPred2, iPred3)
iAggregate = sweep(iAggregate, 2, iMixWeights, '*')
iAggregate = rowSums(iAggregate)
iMSE.test.pr3 = mean((iAggregate - dfData$y[iTest])^2)
plot(iAggregate, dfData$y[iTest], pch=20)
summary(lm(iAggregate ~ dfData$y[iTest]))

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




