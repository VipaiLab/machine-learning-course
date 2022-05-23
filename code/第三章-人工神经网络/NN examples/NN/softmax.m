function Y = softmax(X)
u = exp(X);
Y = u./sum(u);
