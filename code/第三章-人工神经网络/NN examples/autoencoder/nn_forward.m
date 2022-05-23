function nn = nn_forward(nn,batch_x,batch_y)    
    s = size(nn.cost) + 1;%s为cost矩阵的行和列2维向量并各加1,这步配合第51行的nn.cost(s)
                          %实际效果其实就是每次在cost行向量后挤入一个新值
    batch_x = batch_x';
    batch_y = batch_y';
    m = size(batch_x,2);%size（，2）得到矩阵列数
    nn.a{1} = batch_x;
    cost2 = 0;%cost2指cost的第二个和式即添加的正则项
    for k = 2 : nn.depth
        y = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);%repmat(A,m,n)将A复制m×n块
        %由于进行批处理，将m组数据存在矩阵同时处理，而对每组数据来说阈值设定是相同的，故将b复制m次
        %此处y即为所给推导方法中的z.
        if nn.batch_normalization
            nn.E{k-1} = nn.E{k-1}*nn.vecNum + sum(y,2);
            nn.S{k-1} = nn.S{k-1}.^2*(nn.vecNum-1) + (m-1)*std(y,0,2).^2;
            nn.vecNum = nn.vecNum + m;
            nn.E{k-1} = nn.E{k-1}/nn.vecNum;
            nn.S{k-1} = sqrt(nn.S{k-1}/(nn.vecNum-1));
            y = (y - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
            y = nn.Gamma{k-1}*y+nn.Beta{k-1};
        end;
        if k == nn.depth%输出层激活函数选择
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
            switch nn.active_function%隐层激活函数选择
                case 'sigmoid'
                    nn.a{k} = sigmoid(y);
                case 'tanh'
                    nn.a{k} = tanh(y);
                case 'relu'
                    nn.a{k} = max(y,0);
            end
        end
        cost2 = cost2 +  sum(sum(nn.W{k-1}.^2));%正则项计算
    end
    if nn.encoder == 1%此参数为0故可略
        roj = sum(nn.a{2},2)/m;
        nn.cost(s) = 0.5 * sum(sum((nn.a{k} - batch_y).^2))/m + 0.5 * nn.weight_decay * cost2 + 3 * sum(nn.sparsity * log(nn.sparsity ./ roj) + ...
            (1-nn.sparsity) * log((1-nn.sparsity) ./ (1-roj)));
    else
        if strcmp(nn.objective_function,'MSE')
            nn.cost(s) = 0.5 / m * sum(sum((nn.a{k} - batch_y).^2)) + 0.5 * nn.weight_decay * cost2;
        elseif strcmp(nn.objective_function,'Cross Entropy')
            nn.cost(s) = -0.5*sum(sum(batch_y.*log(nn.a{k})))/m + 0.5 * nn.weight_decay * cost2;
        
    end
    
end