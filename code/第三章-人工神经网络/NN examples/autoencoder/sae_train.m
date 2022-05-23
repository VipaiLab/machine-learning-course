function sae = sae_train(sae,option,train_x)
    disp('SAE Training.....');
    sae.encoder = 1;
    sae = nn_train(sae,option,train_x,train_x);
end