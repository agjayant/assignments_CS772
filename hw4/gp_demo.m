clear;
% generate 100 points from uniformly (0,4*pi)
N = 100;
x_train = ??
% generate responses with y = f(x) + eps
% we will assume f is 'sine' function and eps ~ N(0,0.05)
y_train = ??

% define the GP prior with zero mean function and squared exponential
% kernel of the form k(x_i,x_j) = rho^2*exp(-0.5*(x_i - x_j)^2/l^2)
mu = zeros(N,1);
rho = 1;
l = 0.2; % try all values in [0.2,0.5,1,2,10]
K = sq_exp(x_train,x_train,rho,l) + 1e-10*eye(size(x_train, 1));

% draw a random sample of f from the GP prior and plot it
% you may use the provided draw_from_gp.m file (it basically draws from a
% multivariate Gaussian with specific mean and cov. matrix)

??

% compute the GP posterior's mean and plot it
??

% plot the true function as well
??

    
  