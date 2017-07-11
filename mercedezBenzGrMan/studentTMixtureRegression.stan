data {
    int<lower=1> Ntotal; // number of observations
    real y[Ntotal]; // response variable - normally distributed
    int<lower=2> iMixtures; // number of mixture distributions
    int<lower=1> Ncol; // total number of columns in model matrix
    matrix[Ntotal, Ncol] X; // model matrix
}
transformed data{
  real betaShape;
  real betaRate;
  betaShape = 0.5;
  betaRate = 1e-4;
}
parameters { // the parameters to track
    real<lower=0> sigma[iMixtures]; // scale parameters for t distribution  
    simplex[iMixtures] iMixWeights; // weights for the number of mixtures (should sum to one)
    vector[(Ncol-1)] betasMix1; // regression parameters for each mixture component
    vector[(Ncol-1)] betasMix2; // regression parameters for each mixture component
    ordered[iMixtures] mu; // ordered intercept
    real<lower=1> nu[iMixtures]; // degrees of freedom parameter
    real<lower=0.1> betaSigma[iMixtures]; // standard deviation parameter for the joint prior for betas
  }
transformed parameters { // calculated parameters
    vector[Ntotal] muMix1; // number of fitted values
    vector[Ntotal] muMix2; // number of fitted values
    matrix[Ntotal, (Ncol-1)] mX2; // new model matrix without intercept
    mX2 = X[,2:Ncol]; 
    // calculate fitted values without intercept
    muMix1 = mX2 * betasMix1;
    muMix2 = mX2 * betasMix2;
}
model {
  // see stan manual page 187 for an example
  real ps[iMixtures]; // temporary variable for log components
  // any priors go here, get some information from ones we fit earlier
  nu ~ exponential(1/29.0);
  mu[1] ~ normal(80, 10);
  mu[2] ~ normal(110, 10);
  betasMix1 ~ normal(0, betaSigma[1]);
  betasMix2 ~ normal(0, betaSigma[2]);
  // prior for standard deviations of coefficients
  betaSigma ~ gamma(betaShape, betaRate);
  sigma ~ cauchy(0, 2.5); // weak prior
  iMixWeights ~ dirichlet(rep_vector(2.0, iMixtures));
  // loop to calculate likelihood
  for(n in 1:Ntotal){
    // second loop for number of mixture components
    ps[1] = log(iMixWeights[1]) + student_t_lpdf(y[n] | nu[1], mu[1]+muMix1[n], sigma[1]);
    ps[2] = log(iMixWeights[2]) + student_t_lpdf(y[n] | nu[2], mu[2]+muMix2[n], sigma[2]);
    target += log_sum_exp(ps);
  }
}

