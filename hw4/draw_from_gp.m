function f = draw_from_gp(mu,K)
    A = chol(K, 'lower');
    Z = randn(length(mu), 1);
    f = bsxfun(@plus, mu(:), A*Z)';
end