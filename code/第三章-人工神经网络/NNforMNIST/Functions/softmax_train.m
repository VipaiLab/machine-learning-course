function theta = softmax_train(x,y)
    height = size(y,1);
    width = size(x,1);
    theta = rand(height,width);
    output = 1/sum(sum(theta*x))*exp(theta*x);
    ouput = max(output,[],1);
    
end