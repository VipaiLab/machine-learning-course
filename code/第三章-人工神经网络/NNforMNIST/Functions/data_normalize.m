function y = data_normalize(x)
[x, mu, sigma] = zscore(x);
end