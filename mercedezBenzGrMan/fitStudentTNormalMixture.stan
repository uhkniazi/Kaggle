data {
    int<lower=1> Ntotal; // number of observations
    real y[Ntotal]; // response variable - normally distributed
}

parameters { // the parameters to track
    ordered[2] mu; // number of means to track Breaking the Labeling Degeneracy by Enforcing an Ordering
    //real mu[iMixtures]; // number of means to track
    real<lower=0> sigma[2]; // scale parameters for normal distribution  
    real<lower=1> nu;
    simplex[2] iMixWeights; // weights for the number of mixtures (should sum to one)
  }
// transformed parameters {
//   
// }
model {
  // see stan manual page 187 for an example
  real ps[2]; // temporary variable for log components
  // any priors go here 
  nu ~ exponential(1/29.0);
  mu ~ normal(100, 12);
  sigma ~ cauchy(0, 2.5); // weak prior
  iMixWeights ~ dirichlet(rep_vector(2.0, 2));
  // loop to calculate likelihood
  for(n in 1:Ntotal){
    // number of mixture components
    ps[1] = log(iMixWeights[1]) + student_t_lpdf(y[n] | nu, mu[1], sigma[1]);
    ps[2] = log(iMixWeights[2]) + normal_lpdf(y[n] | mu[2], sigma[2]);
    target += log_sum_exp(ps);
  }
}

