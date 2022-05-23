function nn = nn_backpropagation(nn,batch_y)
    batch_y = batch_y';
    m = size(nn.a{1},2);%m为矩阵的列数，即一batch的数目
    nn.theta{1} = 0;%nn.theta即所给推导公式中的δ，特别的下面 nn.theta{nn.depth}对应推导中δ（l）
                    %和之前一样{ }用来记录每一层的δ大小，这也是后向传播的意义
    switch nn.output_function 
        case 'sigmoid'
            nn.theta{nn.depth} = -(batch_y-nn.a{nn.depth}) .* nn.a{nn.depth} .* (1 - nn.a{nn.depth});
        case 'tanh'
            nn.theta{nn.depth} = -(batch_y-nn.a{nn.depth}) .* (1 - nn.a{nn.depth}.^2);
        case 'softmax'
            nn.theta{nn.depth} = nn.a{nn.depth} - batch_y;
    end
    if nn.batch_normalization
        x = nn.W{nn.depth-1} * nn.a{nn.depth-1} + repmat(nn.b{nn.depth-1},1,m);
        x = (x - repmat(nn.E{nn.depth-1},1,m))./repmat(nn.S{nn.depth-1}+0.0001*ones(size(nn.S{nn.depth-1})),1,m);
        temp = nn.theta{nn.depth}.*x;
        nn.Gamma_grad{nn.depth-1} = sum(mean(temp,2));
        nn.Beta_grad{nn.depth-1} = sum(mean(nn.theta{nn.depth},2));
        nn.theta{nn.depth} = nn.Gamma{nn.depth-1}*nn.theta{nn.depth}./repmat((nn.S{nn.depth-1}+0.0001),1,m);
    end;
    %第二项是引入正则项后求导产生的，即正则项系数nn.weight_decay乘以w
    nn.W_grad{nn.depth-1} = nn.theta{nn.depth}*nn.a{nn.depth-1}'/m + nn.weight_decay*nn.W{nn.depth-1};
    nn.b_grad{nn.depth-1} = sum(nn.theta{nn.depth},2)/m;
    %上面式子中sum函数，无参数则累加列得到行向量，sum（，2）则累加行得到列向量
    %这两个式子都对m个梯度求了平均
    switch nn.active_function
        case 'sigmoid'
            if nn.encoder == 0;
                for ll = 2 : nn.depth - 1
                    k = nn.depth - ll + 1;%这里就用到了后向传播的典型意义，从后向前递推，求δ
                    %下式中都将δ写成矩阵形式而不像老师推导中给出的δi，故式子直接写成矩阵相乘
                    nn.theta{k} = ((nn.W{k}'*nn.theta{k+1})) .* nn.a{k} .* (1 - nn.a{k});
                    if nn.batch_normalization%不解释
                        x = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);
                        x = (x - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
                        temp = nn.theta{k}.*x;
                        nn.Gamma_grad{k-1} = sum(mean(temp,2));
                        nn.Beta_grad{k-1} = sum(mean(nn.theta{k},2));
                        nn.theta{k} = nn.Gamma{k-1}*nn.theta{k}./repmat((nn.S{k-1}+0.0001),1,m);
                    end;
                    nn.W_grad{k-1} = nn.theta{k}*nn.a{k-1}'/m + nn.weight_decay*nn.W{k-1};
                    nn.b_grad{k-1} = sum(nn.theta{k},2)/m;%前面已经解释
                end
            else
                roj = sum(nn.a{2},2)/m;
                temp = (-nn.sparsity./roj+(1-nn.sparsity)./(1-roj));
                nn.theta{2} = ((nn.W{2}'*nn.theta{3}) + nn.beta*repmat(temp,1,m)) .* nn.a{2} .* (1 - nn.a{2});
                nn.W_grad{1} = nn.theta{2}*nn.a{1}'/m + nn.weight_decay*nn.W{1};
                nn.b_grad{1} = sum(nn.theta{2},2)/m;
            end
        

            
        case 'tanh'
            for ll = 2 : nn.depth - 1
                if nn.encoder == 0;
                    k = nn.depth - ll + 1;
                    nn.theta{k} = ((nn.W{k}'*nn.theta{k+1})) .* (1-nn.a{k}.^2);
                    if nn.batch_normalization
                        x = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);
                        x = (x - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
                        temp = nn.theta{k}.*x;
                        nn.Gamma_grad{k-1} = sum(mean(temp,2));
                        nn.Beta_grad{k-1} = sum(mean(nn.theta{k},2));
                        nn.theta{k} = nn.Gamma{k-1}*nn.theta{k}./repmat((nn.S{k-1}+0.0001),1,m);
                    end;
                    nn.W_grad{k-1} = nn.theta{k}*nn.a{k-1}'/m + nn.weight_decay*nn.W{k-1};
                    nn.b_grad{k-1} = sum(nn.theta{k},2)/m;
                else
                    roj = sum(nn.a{2},2)/m;
                    temp = (-nn.sparsity./roj+(1-nn.sparsity)./(1-roj));
                    nn.theta{2} = ((nn.W{2}'*nn.theta{3}) + nn.beta*repmat(temp,1,m)) .* (1-nn.a{2}.^2);
                    nn.W_grad{1} = nn.theta{2}*nn.a{1}'/m + nn.weight_decay*nn.W{1};
                    nn.b_grad{1} = sum(nn.theta{2},2)/m;
                end
            end
            
        case 'relu'
            if nn.encoder == 0;
                for ll = 2 : nn.depth - 1
                    k = nn.depth - ll + 1;
                  
                    nn.theta{k} = ((nn.W{k}'*nn.theta{k+1})).*(nn.a{k}>0);
                    if nn.batch_normalization
                        x = nn.W{k-1} * nn.a{k-1} + repmat(nn.b{k-1},1,m);
                        x = (x - repmat(nn.E{k-1},1,m))./repmat(nn.S{k-1}+0.0001*ones(size(nn.S{k-1})),1,m);
                        temp = nn.theta{k}.*x;
                        nn.Gamma_grad{k-1} = sum(mean(temp,2));
                        nn.Beta_grad{k-1} = sum(mean(nn.theta{k},2));
                        nn.theta{k} = nn.Gamma{k-1}*nn.theta{k}./repmat((nn.S{k-1}+0.0001),1,m);
                    end;
                    nn.W_grad{k-1} = nn.theta{k}*nn.a{k-1}'/m + nn.weight_decay*nn.W{k-1};
                    nn.b_grad{k-1} = sum(nn.theta{k},2)/m;
                end
            else
                roj = sum(nn.a{2},2)/m;
                temp = (-nn.sparsity./roj+(1-nn.sparsity)./(1-roj));
                M = max(nn.a{2},0);
                M = M./max(M,0.001);
                    
                nn.theta{2} = ((nn.W{2}'*nn.theta{3}) + nn.beta*repmat(temp,1,m)) .* M;
                nn.W_grad{1} = nn.theta{2}*nn.a{1}'/m + nn.weight_decay*nn.W{1};
                nn.b_grad{1} = sum(nn.theta{2},2)/m;
            end
    end
    
end