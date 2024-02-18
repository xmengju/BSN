functions {
     real signnum(real x) { return x < 0 ? -1 : x > 0; }
     
    matrix theta2gamma(vector theta, int p, int d, matrix II){
     
      int k = 0;
      int I;
      int J; 
      matrix[p,d] Gamma = II;
      matrix[p,p] tmp; 

      for(i in 1:d){
        I = d - i + 1;  // i = 1 -> I = d; i = d; I = 1
        for(j in (I+1):p){
         J = p + (I +1) - j;  // j = i + 1 -> J = p; j = p -> J = i + 1
         k = k + 1;
         tmp = diag_matrix(rep_vector(1, p));
         tmp[I,J] = -sin(theta[k]);
         tmp[J,I] =  sin(theta[k]);
         tmp[I,I] = cos(theta[k]);
         tmp[J,J] =  cos(theta[k]);
         Gamma = tmp*Gamma;
       }
      }
      return Gamma; 
    }

  real myinversecdf(real xx, real uu){
     real cc =  normal_cdf(pi()/2| 0, xx) -  normal_cdf(-pi()/2| 0,xx);
     if(uu<=0.5){
        return( max([-pi()/2,inv_Phi(uu*cc + normal_cdf(-pi()/2|0, xx))*(xx)]));
     }else{
        return(min( [pi()/2, -inv_Phi((1-uu)*cc + normal_cdf(-pi()/2|0,xx))*(xx)]));
     }
   }
  }
  
  data {
      int<lower=0> n;   // number of data items
      int<lower=0> p;   // number of predictors
      int<lower=0> d;   // number of reduced dimensions 
      array[n,p,p] real X;   // predictor matrix
      vector[n] Y;      // outcome vector
      real<lower=0> sigma_prior;  // error scale
      real tau;
  }
  
  transformed data{
      int<lower=0> nl = d*(p-d)+ (d*(d-1)/2);
      matrix[p,d] II = rep_matrix(0, p, d);

      for(i in 1:d){
        II[i,i] = 1;
      } 
  }

  parameters {
      ordered[d] beta;           // coefficients 
      real beta0; // intercept 
      real<lower=0> sigma;  // error scale
      vector<lower=0>[nl] lambda;
      vector<lower=0, upper = 1>[nl] tmp_hs; // inverse CDF sampling 
  }
    
  transformed parameters{
      vector<lower=-pi()/2,upper = pi()/2>[nl] theta;
      matrix[p,d] Gamma;
      
      for(i in 1:nl){
         theta[i] = myinversecdf(tau*lambda[i], tmp_hs[i]);
      }
      
      Gamma =  theta2gamma(theta, p, d, II);
   } 
   
  model{
      vector[n] mu;
      to_vector(lambda) ~ cauchy(0, 1);
      tmp_hs ~ uniform(0, 1);
      beta0 ~ normal(0, 1);  
      beta ~ normal(0,5);  

      sigma ~ exponential(log(2)/sigma_prior); 
      
      for(i in 1:n){
         mu[i]  = dot_product(diagonal(Gamma'*to_matrix(X[i])*Gamma), beta) + beta0;
      }
  
      Y ~ normal(mu, sigma);  
  }

  