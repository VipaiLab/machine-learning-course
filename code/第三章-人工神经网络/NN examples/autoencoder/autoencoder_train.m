function autoencoder = autoencoder_train(autoencoder,train_x,train_y)
    option.batch_size = 100;
    option.iteration = 10;
    for k = 1 : numel(autoencoder.size)- 2
        disp(['The ' num2str(k) '/' num2str(numel(autoencoder.size)-1) ' hidden layer is traing']);
        sae = sae_create([autoencoder.size(k),autoencoder.size(k+1)]);
        sae = sae_train(sae,option,train_x);
        autoencoder.W{k} = sae.W{1};
        autoencoder.b{k} = sae.b{1};
        sae = nn_predict(sae,train_x);
        train_x = sae.a{2}';
    end
    k = k + 1;
    disp(['The ' num2str(k) '/' num2str(numel(autoencoder.size)-1) ' hidden layer is traing']);
    nn = nn_create([autoencoder.size(k),autoencoder.size(k+1)]);
    nn = nn_train(nn,option,train_x,train_y);
    autoencoder.W{k} = nn.W{1};
    autoencoder.b{k} = nn.b{1};
end