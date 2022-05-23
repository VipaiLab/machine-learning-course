function nn = nn_predict(nn,batch_x)%前面文件中都注释过
    batch_x = batch_x';
    m = size(batch_x,2);
    nn.a{1} = batch_x;
    for k = 2 : nn.depth
        y = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);
        if nn.batch_normalization
            y = (y - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
            y = nn.Gamma{k-1}*y+nn.Beta{k-1};
        end;
        if k == nn.depth
            switch nn.output_function
                case 'sigmoid'
                    nn.a{k} = sigmoid(y);
                case 'tanh'
                    nn.a{k} = tanh(y);
                case 'relu'
                    nn.a{k} = max(y,0);
                case 'softmax'
                    nn.a{k} = softmax(y);
            end
        else
            switch nn.active_function
                case 'sigmoid'
                    nn.a{k} = sigmoid(y);
                case 'tanh'
                    nn.a{k} = tanh(y);
                case 'relu'
                    nn.a{k} = max(y,0);
            end
        end
    end
end