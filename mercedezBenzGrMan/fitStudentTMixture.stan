data {
    int<lower=1> Ntotal; // number of observations
    real y[Ntotal]; // response variable - normally distributed
    int<lower=2> iMixtures; // number of mixture distributions
}

parameters { // the parameters to track
    ordered[iMixtures] mu; // number of means to track Breaking the Labeling Degeneracy by Enforcing an Ordering
    //real mu[iMixtures]; // number of means to track
    real<lower=0> sigma[iMixtures]; // scale parameters for normal distribution  
    real<lower=1> nu[iMixtures];
    simplex[iMixtures] iMixWeights; // weights for the number of mixtures (should sum to one)
  }
// transformed parameters {
//   
// }
model {
  // see stan manual page 187 for an example
  real ps[iMixtures]; // temporary variable for log components
  // any priors go here 
  nu ~ exponential(1/29.0);
  mu ~ normal(100, 12);
  sigma ~ cauchy(0, 2.5); // weak prior
  iMixWeights ~ dirichlet(rep_vector(2.0, iMixtures));
  // loop to calculate likelihood
  for(n in 1:Ntotal){
    // second loop for number of mixture components
    for (k in 1:iMixtures){
      ps[k] = log(iMixWeights[k]) + student_t_lpdf(y[n] | nu[k], mu[k], sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}

