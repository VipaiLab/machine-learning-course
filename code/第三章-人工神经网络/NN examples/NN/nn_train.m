function nn = nn_train(nn,option,train_x,train_y)
    iteration = option.iteration;%迭代次数，即操作几轮
    batch_size = option.batch_size;%累积BP算法中的批量大小
    m = size(train_x,1);%size(，1)返回矩阵行数
    num_batches = m / batch_size;
    for k = 1 : iteration
        kk = randperm(m);%打乱1到m的顺序 
        if  strcmp(nn.optimization_method,'Adam')
            nn.AdamTime = nn.AdamTime+1;
        end;
        for l = 1 : num_batches %即对60000个训练数据打乱后打成小包
            batch_x = train_x(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            batch_y = train_y(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            nn = nn_forward(nn,batch_x,batch_y); %前向计算
            nn = nn_backpropagation(nn,batch_y); %后向计算
            nn = nn_applygradient(nn);           %梯度下降
        end
       %disp(['Iteration ' num2str(k) '/' num2str(iteration) ' : ' num2str(t) ' seconds']);
    end
    %figure;
    %plot(nn.cost);grid on;
end