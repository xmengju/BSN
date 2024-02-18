###-- An example running Bayesian scalar-on-network regression (BSN)
rm(list = ls())
source("function.R")
library(expm)
library(pracma)
library(cmdstanr)
library(bayesplot)

## generate predictor variable 
n <- 200; p <- 5; d <- 2
Ms <- array(NA, dim = c(n, p, p)) 
for(i in 1:n){
  U <- randortho(p) 
  Ms[i,,] <- U%*%diag(runif(p, 1, 2))%*%t(U) 
}

## generate Gamma matrix and B
set.seed(1)
nl <- p*d - d^2/2 - d/2
theta <- runif(nl, -pi/2, pi/2)
theta[sample(1:nl, 0.5*round(nl))] <- 0
Gamma <- theta_to_gamma(theta, p, d)
B <- diag(c(1,2))
Gamma[,2] <- -Gamma[,2] # adjust the sign

## find mapped matrices in the tangent space
M_ast_inv_square_root <-  matrixsqrtinv(apply(Ms, c(2,3), mean))
X_tan_log <- array(NA, dim = c(n, p, p)) 
for(i in 1:n){
  X_tan_log[i,,] <- logm(M_ast_inv_square_root %*% Ms[i,,]%*%M_ast_inv_square_root)
}

## generate response
y <- rep(NA, n)
for(i in 1:n){
  y[i] <- sum ((t(Gamma)%*%X_tan_log[i,,]%*%Gamma) * B)
}
y <- y + rnorm(n, 0, 0.1)

## run mcmc for linear model without/with regularization
bsn_t <- cmdstan_model("BSN-N(S).stan")
bsn_ts <- cmdstan_model("BSN-T(S).stan")

sigma_prior <- 1 # in practice from some initial linear fit using vectorized matrix predictors
data.train <- list(n = n, p = p, d = d, X = X_tan_log, Y = y, sigma_prior = sigma_prior)
chains  <- 1; parallel_chains <- 1; refresh <- 50; iter_warmup <- 600; iter_sampling <- 300
fit_1 <- bsn_t$sample(data = data.train, chains = chains, parallel_chains = parallel_chains, 
             refresh = refresh, iter_warmup = iter_warmup, iter_sampling  = iter_sampling)
data.train <- list(n = n, p = p, d = d, X = X_tan_log, Y = y, sigma_prior = sigma_prior, tau = 0.1)
fit_2 <- bsn_ts$sample(data = data.train, chains = chains, parallel_chains = parallel_chains, 
                      refresh = refresh, iter_warmup = iter_warmup, iter_sampling  = iter_sampling)

## adjust the sign of each column of sampled Gamma; forcing the entry in the sampled Gamma with the largest absolute value
## in each column of Gamma to be positive 
Gamma_fit1 <- process_gamma(fit_1$draws("Gamma"), Gamma)  
Gamma_fit2 <- process_gamma(fit_2$draws("Gamma"), Gamma)  

mcmc_intervals(Gamma_fit1)
mcmc_intervals(fit_1$draws("beta"))

mcmc_intervals(Gamma_fit2)
mcmc_intervals(fit_2$draws("beta"))

## make predictions (in sample)
pred_1 <- pred.lm(fit_1, X = X_tan_log, X_new = X_tan_log) # array of dimension (iter_sampling, nchains, n)
pred_2 <- pred.lm(fit_2, X = X_tan_log, X_new = X_tan_log) 

plot(apply(pred_1, 3, median), y, xlab = "pred y")
abline(a = 0, b = 1, col = "red")
plot(apply(pred_2, 3, median), y,  xlab = "pred y")
abline(a = 0, b = 1, col = "red")


