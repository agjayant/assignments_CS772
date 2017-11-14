function K = sq_exp(X1,X2,sigma,l)
% X1 (N1xD) and X2 (N2xD) are two sets of points
% This function will compute an N1 x N2 matrix of their pairwise
% similarities using a squared exponential kernel
    N1 = size(X1,1);
    N2 = size(X2,1);
    for n=1:N1
        for m=1:N2
            sqdist = (X1(n,:)-X2(m,:))*(X1(n,:)-X2(m,:))';
            K(n,m) = sigma^2*exp(-sqdist/l^2);
        end
    end
end