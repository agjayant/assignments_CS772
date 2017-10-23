clear;

% load the data (documents x words matrix, N documents, M words)
load('20newsgroups');
[N M] = size(X);

% hyperparameters of the gamma priors on U,V,Lambda (don't change these)
% these choice of hyperparams leads to an uninformative/vague gamma prior
a_u = 1;b_u = 1e-6;
a_v = 1;b_v = 1e-6;

% number of latent factors
K = 20;

% initialization
U = gamrnd(a_u,1/b_u,N,K); % NxK matrix, drawn randomly from the prior
V = gamrnd(a_v,1/b_v,M,K); % MxK matrix, drawn randomly from the prior

% number of iterations (you can play with these in experimentation)
max_iters = 1000;
burnin = 500;

% prepare the training and test data

[I J vals] = find(X); % all nonzero entries (with indices) in X
% vals is a vector containing denotes the nonzero entries in X
% I and J are vectors with row and column indices resepectively

% Do an 80/20 split of matrix entries (nonzeros) into training and test data
num_elem = length(I);
rp = randperm(num_elem);
num_nz = num_elem;

% indices of training set entries
I_train = I(rp(1:round(0.8*num_elem)));
J_train = J(rp(1:round(0.8*num_nz)));

% indices of test set entries
J_test = J(rp(round(0.8*num_nz)+1:end));
I_test = I(rp(round(0.8*num_elem)+1:end));

% values of the training and test entries
vals_train = vals(rp(1:round(0.8*num_nz)));
vals_test = vals(rp(round(0.8*num_nz)+1:end));

% run the Gibbs sampler for max_iters iterations
% in each iteration, draw samples of U, V, lambda
% Note: you may have to draw samples of some additional latent variables
% (your solution to Problem 1 should tell you what these variables will be)

U_avg = zeros(N,K);
V_avg = zeros(M,K);

maeF1 = fopen('mae.txt','w');
maeF2 = fopen('mae_avg.txt','w');

for iters=1:max_iters
    
    
    eta = zeros(round(0.8*num_nz), K);
    for i=1:round(0.8*num_nz)
        eta(i,:) = U(I_train(i),:).*V(J_train(i),:)/(U(I_train(i),:)*V(J_train(i),:)');
    end
    
    X_mnk = mnrnd(vals_train,eta );
        
    % Sample U using its local conditional posterior
    % Note: wherever possible, try to vectorize the code for more efficiency
    % e.g., instead  of looping over each entry of U, you can draw one 
    %row at a  time
    
    X_ndk = zeros(N,K);    
    for i=1:N
        n_ind = find(I_train == i);
        X_ndk(i,:) = sum(X_mnk(n_ind,:)) + a_u;
    end
    
    V_dk = repmat(sum(V),N,1) + b_u;
    V_dk = arrayfun(@(a) 1/a, V_dk);
    
    U = gamrnd(X_ndk, V_dk);           
    
    % Sample V using its local conditional posterior
    % Note: wherever possible, try to vectorize the code for more efficiency
    % e.g., instead  of looping over each entry of V, you can draw one 
    % row at a  time
    
    X_dmk = zeros(M,K);    
    for i=1:M
        m_ind = find(J_train == i);
        X_dmk(i,:) = sum(X_mnk(m_ind,:)) + a_v;
    end
    
    U_dk = repmat(sum(U),M,1) + b_v;
    U_dk = arrayfun(@(a) 1/a, U_dk);
    
    V = gamrnd(X_dmk, U_dk); 
    
    % Calculate the reconstructor error (mean absolute error) of training
    % and test entries. You need to do this using two ways: (1) Using the 
    % samples of U, V, lambda from this iteration (2) Using Monte Carlo 
    % averaging; the latter only has to be done after the burn-in period
    
    % Approach 1 (using current samples of U and V from this iteration) 
    mae_train = 0 ;
    for i=1:round(0.8*num_nz)
        mae_train = mae_train + abs(U(I_train(i),:)*V(J_train(i),:)' - vals_train(i));
    end
    mae_train = mae_train/round(0.8*num_nz);
    
    mae_test = 0;
    for i=1:round(0.2*num_nz)
        mae_test = mae_test + abs(U(I_test(i),:)*V(J_test(i),:)' - vals_test(i));
    end
    mae_test = mae_test/round(0.2*num_nz);
    fprintf('Done with iteration %d, MAE_train = %f, MAE_test = %f\n',iters,mae_train,mae_test);
    fprintf(maeF1, '%d %f %f\n',iters,mae_train,mae_test);
    
    % Approach 2 (using Monte Carlo averaging; but only using the
    % post-burnin samples of U and V)
    if iters > burnin
        U_avg = (U_avg*(iters-burnin-1) + U)/(iters-burnin);
        V_avg = (V_avg*(iters-burnin-1) + V)/(iters-burnin);
        
        mae_train_avg = 0;
        for i=1:round(0.8*num_nz)
            mae_train_avg = mae_train_avg + abs(U_avg(I_train(i),:)*V_avg(J_train(i),:)' - vals_train(i));
        end
        mae_train_avg = mae_train_avg/round(0.8*num_nz);
        
        mae_test_avg = 0;
        for i=1:round(0.2*num_nz)
            mae_test_avg = mae_test_avg + abs(U_avg(I_test(i),:)*V_avg(J_test(i),:)' - vals_test(i));
        end
        mae_test_avg = mae_test_avg/round(0.2*num_nz);
        
        fprintf('With Posterior Averaging, MAE_train_avg = %f, MAE_test_avg = %f\n',mae_train_avg,mae_test_avg);
        fprintf(maeF2, '%d %f %f\n',iters, mae_train_avg,mae_test_avg);
    end
end

fclose(maeF1);
fclose(maeF2);

% Finally, let's print the K topics (top 20 words from each topic)
% For this, you need to take the V matrix and finds the 20 largest entries in
% each column of V. The function 'printtopics' already does it but if you
% are implementing in some other language, you will have to write this
% function by yourself, using the same logic)
topicsF1 = fopen('topics_result.txt','w');
printtopics(V,topicsF1);
fclose(topicsF1);

topicsF2 = fopen('topics_result_avg.txt','w');
printtopics(V_avg,topicsF2);
fclose(topicsF2);
