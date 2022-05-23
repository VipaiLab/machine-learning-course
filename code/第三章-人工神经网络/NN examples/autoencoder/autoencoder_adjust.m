function autoencoder = autoencoder_adjust(autoencoder,train_x,train_y)
    disp('The cnn is adjusting');
    iteration = 2000;
    batch_size = 100;
    m = size(train_x,1);
    num_batches = m / batch_size;
    autoencoder.learning_rate = 1.0;
    for k = 1 : iteration
        tic;
        kk = randperm(m);
        for l = 1 : num_batches
            batch_x = train_x(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            batch_y = train_y(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            autoencoder = nn_forward(autoencoder,batch_x,batch_y);
            autoencoder = nn_backpropagation(autoencoder,batch_y);
            autoencoder = nn_applygradient(autoencoder);
        end
        t = toc;
        disp(['Iteration ' num2str(k) '/' num2str(iteration) ' : ' num2str(t) ' seconds']);
    end
    figure;
    plot(autoencoder.cost);grid on;
end