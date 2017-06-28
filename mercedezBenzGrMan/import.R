# File: import.R
# Auth: uhkniazi
# Date: 27/06/2017
# Desc: import data and decide on the distribution function


library(LearnBayes)

set.seed(123) # for replication

dfData = read.csv('mercedezBenzGrMan/DataExternal/train.csv', header=T)
dim(dfData)
head(dfData)

ivTime = dfData$y
summary(ivTime)
sd(ivTime)

## define a log posterior function
lp = function(theta, data){
  # we define the sigma on a log scale as optimizers work better
  # if scale parameters are well behaved
  s = exp(theta[2])
  m = theta[1]
  d = data$vector # observed data vector
  log.lik = sum(dnorm(d, m, s, log=T))
  log.prior = 1
  log.post = log.lik + log.prior
  return(log.post)
}

# sanity check for function
# choose a starting value
start = c('mu'=mean(ivTime), 'sigma'=log(sd(ivTime)))
lData = list('vector'=ivTime)
lp(start, lData)

op = optim(start, lp, control = list(fnscale = -1), data=lData)
op$par
exp(op$par[2])

## try the laplace function from LearnBayes
fit = laplace(lp, start, lData)
fit
se = sqrt(diag(fit$var))
se
fit$mode+1.96*se
fit$mode-1.96*se

# taking the sample
tpar = list(m=fit$mode, var=fit$var*2, df=4)
muSample2.op = sir(lp, tpar, 1000, lData)

sigSample.op = muSample2.op[,'sigma']
muSample.op = muSample2.op[,'mu']

### if we look at the histogram of the time
hist(ivTime, xlab='Response', main='', breaks=50)
plot(density(ivTime))
## we can see the outlier measurements and the normal model is inappropriate for this problem

## sample 4209 values, 20 times, each time drawing a fresh draw of sd and mean from the joint posterior
mDraws = matrix(NA, nrow = 4209, ncol=20)

for (i in 1:20){
  p = sample(1:1000, size = 1)
  s = exp(sigSample.op[p])
  m = muSample.op[p]
  mDraws[,i] = rnorm(4209, m, s)
}

p.old = par(mfrow=c(3, 3))
garbage = apply(mDraws, 2, function(x) hist(x, main='', xlab='', ylab=''))
hist(ivTime, xlab='Speed of light measurements', main='')

## calculate bayesian p-value for this test statistic
getPValue = function(Trep, Tobs){
  left = sum(Trep <= Tobs)/length(Trep)
  right = sum(Trep >= Tobs)/length(Trep)
  return(min(left, right))
}
## define some test quantities to measure the lack of fit
## define a test quantity T(y, theta)
# The procedure for carrying out a posterior predictive model check requires specifying a test
# quantity, T (y) or T (y, θ), and an appropriate predictive distribution for the replications
# y rep [Gelman 2008]
## variance
T1_var = function(Y) return(var(Y))
## is the model adequate except for the extreme tails
T1_symmetry = function(Y, th){
  Yq = quantile(Y, c(0.90, 0.10))
  return(abs(Yq[1]-th) - abs(Yq[2]-th))
} 

## min quantity
T1_min = function(Y){
  return(min(Y))
} 

## max quantity
T1_max = function(Y){
  return(max(Y))
} 

## mean quantity
T1_mean = function(Y){
  return(mean(Y))
} 

## mChecks
mChecks = matrix(NA, nrow=5, ncol=3)
rownames(mChecks) = c('Variance', 'Symmetry', 'Max', 'Min', 'Mean')
colnames(mChecks) = c('Normal', 'NormalCont', 'T')
########## simulate 200 test quantities
mDraws = matrix(NA, nrow = length(ivTime), ncol=200)
mThetas = matrix(NA, nrow=200, ncol=2)
colnames(mThetas) = c('mu', 'sd')

for (i in 1:200){
  p = sample(1:1000, size = 1)
  s = exp(sigSample.op[p])
  m = muSample.op[p]
  mDraws[,i] = rnorm(length(ivTime), m, s)
  mThetas[i,] = c(m, s)
}

mDraws.norm = mDraws
### get the test quantity from the test function
t1 = apply(mDraws, 2, T1_var)
par(p.old)
hist(t1, xlab='Test Quantity - Variance (Normal Model)', main='', breaks=50)
abline(v = var(lData$vector), lwd=2)
mChecks['Variance', 1] = getPValue(t1, var(lData$vector))
# 0.48, the result from Figure 6.4 Gelman [2013]
# The sample variance does not make a good test statistic because it is a sufficient statistic of
# the model and thus, in the absence of an informative prior distribution, the posterior
# distribution will automatically be centered near the observed value. We are not at all
# surprised to find an estimated p-value close to 1/2 . [Gelman 2008]

## test for symmetry
t1 = sapply(seq_along(1:200), function(x) T1_symmetry(mDraws[,x], mThetas[x,'mu']))
t2 = sapply(seq_along(1:200), function(x) T1_symmetry(lData$vector, mThetas[x,'mu']))
plot(t2, t1, xlim=c(-12, 12), ylim=c(-12, 12), pch=20, xlab='Realized Value T(Yobs, Theta)',
     ylab='Test Value T(Yrep, Theta)', main='Symmetry Check (Normal Model)')
abline(0,1)
mChecks['Symmetry', 1] = getPValue(t1, t2) # we should see somewhere around 0.1 to 0.2 on repeated simulations
# The estimated p-value is 0.26, implying that any observed asymmetry in the middle of the distribution can easily be
# explained by sampling variation. [Gelman 2008]

## testing for outlier detection i.e. the minimum value show in the histograms earlier
t1 = apply(mDraws, 2, T1_min)
t2 = T1_min(lData$vector)
mChecks['Min',1] = getPValue(t1, t2)

## maximum value
t1 = apply(mDraws, 2, T1_max)
t2 = T1_max(lData$vector)
mChecks['Max', 1] = getPValue(t1, t2)

## mean value
t1 = apply(mDraws, 2, T1_mean)
t2 = T1_mean(lData$vector)
mChecks['Mean', 1] = getPValue(t1, t2)


########## try a contaminated normal distribution
lp2 = function(theta, data){
  # we define the sigma on a log scale as optimizers work better
  # if scale parameters are well behaved
  s = exp(theta[2])
  m = theta[1]
  mix = 0.95
  cont = theta[3]
  d = data$vector # observed data vector
  log.lik = sum(log(dnorm(d, m, s) * mix + dnorm(d, m, s*cont) * (1-mix)))
  log.prior = 1
  log.post = log.lik + log.prior
  return(log.post)
}

# sanity check for function
# choose a starting value
start = c('mu'=mean(ivTime), 'sigma'=log(sd(ivTime)), 'cont'=1)
lp2(start, lData)

op = optim(start, lp2, control = list(fnscale = -1), data=lData)
op$par
exp(op$par[2])

## try the laplace function from LearnBayes
fit2 = laplace(lp2, start, lData)
fit2
se2 = sqrt(diag(fit2$var))
se2
fit2$mode+1.96*se2
fit2$mode-1.96*se2

# taking the sample
tpar = list(m=fit2$mode, var=fit2$var*2, df=4)
muSample2.op = sir(lp2, tpar, 1000, lData)

sigSample.op = muSample2.op[,'sigma']
muSample.op = muSample2.op[,'mu']
contSample.op = muSample2.op[,'cont']

########## simulate 200 test quantities
mDraws = matrix(NA, nrow = length(ivTime), ncol=200)
mThetas = matrix(NA, nrow=200, ncol=2)
colnames(mThetas) = c('mu', 'sd')

for (i in 1:200){
  p = sample(1:1000, size = 1)
  s = exp(sigSample.op[p])
  m = muSample.op[p]
  co = contSample.op[p]
  ## this will take a sample from a contaminated normal distribution
  sam = function() {
    ind = rbinom(1, 1, 0.95)
    return(ind * rnorm(1, m, s) + (1-ind) * rnorm(1, m, s*co))
  }
  mDraws[,i] = replicate(length(ivTime), sam())
  mThetas[i,] = c(m, s)
}

mDraws.normCont = mDraws
## get the p-values for the test statistics
t1 = apply(mDraws, 2, T1_var)
mChecks['Variance', 2] = getPValue(t1, var(lData$vector))

## test for symmetry
t1 = sapply(seq_along(1:200), function(x) T1_symmetry(mDraws[,x], mThetas[x,'mu']))
t2 = sapply(seq_along(1:200), function(x) T1_symmetry(lData$vector, mThetas[x,'mu']))
plot(t2, t1, xlim=c(-12, 12), ylim=c(-12, 12), pch=20, xlab='Realized Value T(Yobs, Theta)',
     ylab='Test Value T(Yrep, Theta)', main='Symmetry Check (Normal Contaminated)')
abline(0,1)
mChecks['Symmetry', 2] = getPValue(t1, t2) 

## testing for outlier detection i.e. the minimum value show in the histograms earlier
t1 = apply(mDraws, 2, T1_min)
t2 = T1_min(lData$vector)
mChecks['Min', 2] = getPValue(t1, t2)

## maximum value
t1 = apply(mDraws, 2, T1_max)
t2 = T1_max(lData$vector)
mChecks['Max', 2] = getPValue(t1, t2)

## mean value
t1 = apply(mDraws, 2, T1_mean)
t2 = T1_mean(lData$vector)
mChecks['Mean', 2] = getPValue(t1, t2)


############################### fit a model using stan to estimate mixture parameters
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
i = sample(1:length(ivTime), size = 300)
stanDso = rstan::stan_model(file='mercedezBenzGrMan/fitNormalMixture.stan')
lStanData = list(Ntotal=length(ivTime[i]), y=ivTime[i], iMixtures=3)
fit.stan = sampling(stanDso, data=lStanData, iter=1000, chains=4, cores=4)
print(fit.stan, digi=3)









######################### try a third distribution, t with a low degrees of freedom
lp3 = function(theta, data){
  # function to use to use scale parameter
  ## see here https://grollchristian.wordpress.com/2013/04/30/students-t-location-scale/
  dt_ls = function(x, df, mu, a) 1/a * dt((x - mu)/a, df)
  ## likelihood function
  lf = function(dat, nu, pred, sigma){
    return(dt_ls(dat, nu, pred, sigma))
  }
  nu1 = exp(theta['nu1']) ## normality parameter for t distribution
  nu2 = exp(theta['nu2']) ## normality parameter for t distribution
  sigma1 = exp(theta['sigma1']) # scale parameter for t distribution
  sigma2 = exp(theta['sigma2']) # scale parameter for t distribution
  m1 = theta['mu1']
  m2 = theta['mu2']
  mix = 0.5# logit.inv(theta['mix'])
  d = data$vector # observed data vector
  if (nu1 < 1 || nu2 < 1) return(-Inf)
  log.lik = sum(log(lf(d, nu1, m1, sigma1) * mix + lf(d, nu2, m2, sigma2) * (1-mix)))
  log.prior = 1 + dunif(nu1, 1, 10, log=T) + dunif(nu2, 1, 10, log=T)
  log.post = log.lik + log.prior
  return(log.post)
}

# sanity check for function
library(numDeriv)
library(car)
logit.inv = function(p) {exp(p)/(exp(p)+1) }
# choose a starting value
start = c('mu1'=mean(ivTime), 'mu2'=mean(ivTime), 'sigma1'=log(sd(ivTime)), 
          'sigma2'=log(sd(ivTime)), 'nu1'=log(2), 'nu2'=log(2))#, 'mix'=logit(0.5))
lp3(start, lData)

op = optim(start, lp3, control = list(fnscale = -1), data=lData)
op$par
logit.inv(op$par['mix'])

mylaplace = function (logpost, mode, data) 
{
  options(warn = -1)
  fit = optim(mode, logpost, gr = NULL,  
              control = list(fnscale = -1, maxit=10000), method='Nelder-Mead', data=data)
  # calculate hessian
  fit$hessian = (hessian(logpost, fit$par, data=data))
  colnames(fit$hessian) = names(mode)
  rownames(fit$hessian) = names(mode)
  options(warn = 0)
  mode = fit$par
  h = -solve(fit$hessian)
  stuff = list(mode = mode, var = h, converge = fit$convergence == 
                 0)
  return(stuff)
}


## try the laplace function from LearnBayes
fit3 = mylaplace(lp3, start, lData)
fit3
se3 = sqrt(diag(fit3$var))

# taking the sample
tpar = list(m=fit3$mode, var=fit3$var*2, df=4)
muSample2.op = sir(lp3, tpar, 10000, lData)

########## simulate 200 test quantities
mDraws = matrix(NA, nrow = length(ivTime), ncol=200)
# mThetas = matrix(NA, nrow=200, ncol=3)
# colnames(mThetas) = c('mu', 'sd', 'nu')

rt_ls <- function(n, df, mu, a) rt(n,df)*a + mu

for (i in 1:200){
  p = sample(1:1000, size = 1)
  ## this will take a sample from a contaminated normal distribution
  sam = function() {
    ind = rbinom(1, 1, 0.5)
    return(ind * rt_ls(1, exp(muSample2.op[p,'nu1']), muSample2.op[p,'mu1'], exp(muSample2.op[p,'sigma1'])) +
             (1-ind) * rt_ls(1, exp(muSample2.op[p,'nu2']), muSample2.op[p,'mu2'], exp(muSample2.op[p,'sigma2'])))
  }
  mDraws[,i] = replicate(length(ivTime), sam())
}

mDraws.t = mDraws
## get the p-values for the test statistics
t1 = apply(mDraws, 2, T1_var)
mChecks['Variance', 3] = getPValue(t1, var(lData$vector))

## test for symmetry
t1 = sapply(seq_along(1:200), function(x) T1_symmetry(mDraws[,x], mThetas[x,'mu']))
t2 = sapply(seq_along(1:200), function(x) T1_symmetry(lData$vector, mThetas[x,'mu']))
plot(t2, t1, xlim=c(-12, 12), ylim=c(-12, 12), pch=20, xlab='Realized Value T(Yobs, Theta)',
     ylab='Test Value T(Yrep, Theta)', main='Symmetry Check (T Distribution)')
abline(0,1)
mChecks['Symmetry', 3] = getPValue(t1, t2) 

## testing for outlier detection i.e. the minimum value show in the histograms earlier
t1 = apply(mDraws, 2, T1_min)
t2 = T1_min(lData$vector)
mChecks['Min', 3] = getPValue(t1, t2)

## maximum value
t1 = apply(mDraws, 2, T1_max)
t2 = T1_max(lData$vector)
mChecks['Max', 3] = getPValue(t1, t2)

## mean value
t1 = apply(mDraws, 2, T1_mean)
t2 = T1_mean(lData$vector)
mChecks['Mean', 3] = getPValue(t1, t2)

mChecks

