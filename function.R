## Givens representation: theta to Gamma
theta_to_gamma <- function(theta, p, d){
  
  tmp <-  diag(1, nrow = p)
  tmp <- tmp[,1:d] 
  k <- 0
  for(i in d:1){
    if ((i+1) <=p){
    for(j in p:(i+1)){
        k <- k+1
        tmp_k<- diag(1, nrow = p)
        tmp_k[i,i] <-  tmp_k[j,j] <- cos(theta[k])
        tmp_k[i,j] <-  -sin(theta[k])
        tmp_k[j,i] <-  sin(theta[k])
        tmp  <- tmp_k%*%tmp
      }
    }
  }
  return(tmp)
}

compute.index <- function(gamma, beta, X){
  
  tmp <- NULL
  if(length(dim(X)) == 3){
    for(i in 1:dim(X)[1]){
      tmp <- c(tmp, sum(diag(t(gamma)%*%X[i,,]%*%gamma)*beta)) 
    }
  }else{
    tmp <- sum(diag(t(gamma)%*%X%*%gamma)*beta)
  }
  return(tmp)
}


pred.lm <- function(fit_lm, X, X_new){
  
  K <- fit_lm$num_chains()
  nsamples <- dim(fit_lm$draws("Gamma"))[1]
  p <- dim(X)[3]
  Gammas <- fit_lm$draws("Gamma")
  d <- length(Gammas[1,1,])/p
  betas <- fit_lm$draws("beta")
  beta0s <- fit_lm$draws("beta0")
  y_pred <- array(NA, dim = c(nsamples, K, nrow(X_new)))
  
  for(i in 1:nsamples){
    for(kk in 1:K){
      for(j in 1:nrow(X_new)){
        y_pred[i,kk, j] <- compute.index(gamma =matrix(Gammas[i,kk,], p, d), matrix(betas[i,kk,], ncol =1), X_new[j,,]) + as.numeric(beta0s[i, kk, ])
      }
    }
  }
  return(y_pred)
}

matrixsqrtinv <-function (S, tol = sqrt(.Machine$double.eps)){
  s <- svd(S)
  nz <- s$d > tol
  S12 = s$u[, nz] %*% diag(1/sqrt(s$d[nz])) %*% t(s$v[, nz])
  return(S12)
}


process_gamma <- function(Gamma_sample, Gamma){
  
  d <- ncol(Gamma)
  p <- nrow(Gamma)
  nsample <-dim(Gamma_sample)[1]
  nchains <- dim(Gamma_sample)[2]
  for(i in 1:d){
    idx <- which.max(abs(Gamma[,i]))
    for(j in 1:nsample){
      for(k in 1:nchains){
        Gamma_sample[j,k, ((i-1)*(p)+ 1):(i*p)] <- Gamma_sample[j,k, ((i-1)*(p)+ 1):(i*p)]* as.numeric(sign(Gamma_sample[j,k, (i-1)*p+idx]))
      }
    }
  }
  return(Gamma_sample)
}
