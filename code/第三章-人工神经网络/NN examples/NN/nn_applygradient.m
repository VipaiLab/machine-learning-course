function nn = nn_applygradient(nn)%都是各种优化方法，我们选normal

if strcmp(nn.optimization_method, 'AdaGrad') || strcmp(nn.optimization_method, 'RMSProp') || strcmp(nn.optimization_method, 'Adam')
    grad_squared = 0;
    if nn.batch_normalization == 0
        for k = 1 : nn.depth-1
            grad_squared = grad_squared + sum(sum(nn.W_grad{k}.^2))+sum(nn.b_grad{k}.^2);
        end;
    else
        for k = 1 : nn.depth-1
            grad_squared = grad_squared + sum(sum(nn.W_grad{k}.^2))+sum(nn.b_grad{k}.^2)+nn.Gamma{k}^2 + nn.Beta{k}^2;
        end;
    end;
end;

for k = 1 : nn.depth-1
    if nn.batch_normalization == 0
        if strcmp(nn.optimization_method, 'normal')
            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k};
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k};
        elseif strcmp(nn.optimization_method, 'AdaGrad')
            nn.rW{k} = nn.rW{k} + nn.W_grad{k}.^2;
            nn.rb{k} = nn.rb{k} + nn.b_grad{k}.^2;
            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k}./(sqrt(nn.rW{k})+0.001);
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k}./(sqrt(nn.rb{k})+0.001);
        elseif strcmp(nn.optimization_method, 'Momentum')
            rho = 0.1;%rho = 0.1;
            nn.vW{k} = rho*nn.vW{k} - nn.learning_rate*nn.W_grad{k};
            nn.vb{k} = rho*nn.vb{k} - nn.learning_rate*nn.b_grad{k};
            nn.W{k} = nn.W{k} + nn.vW{k};
            nn.b{k} = nn.b{k} + nn.vb{k}; 

        elseif strcmp(nn.optimization_method, 'RMSProp')
            rho = 0.9; %rho=0.9
            nn.rW{k} = rho*nn.rW{k} + 0.1*nn.W_grad{k}.^2;
            nn.rb{k} = rho*nn.rb{k} + 0.1*nn.b_grad{k}.^2;

            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k}./(sqrt(nn.rW{k})+0.001);
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k}./(sqrt(nn.rb{k})+0.001); %rho = 0.9
        elseif strcmp(nn.optimization_method, 'Adam')
            rho1 = 0.9;
            rho2 = 0.999;
            nn.sW{k} = rho1*nn.sW{k} + (1-rho1)*nn.W_grad{k};
            nn.sb{k} = rho1*nn.sb{k} + (1-rho1)*nn.b_grad{k};
            nn.rW{k} = rho2*nn.rW{k} + (1-rho2)*nn.W_grad{k}.^2;
            nn.rb{k} = rho2*nn.rb{k} + (1-rho2)*nn.b_grad{k}.^2;
            
            newS = nn.sW{k}/(1-rho1^nn.AdamTime);
            newR = nn.rW{k}/(1-rho2^nn.AdamTime);
            nn.W{k} = nn.W{k} - nn.learning_rate*newS./sqrt(newR+0.00001);
            newS = nn.sb{k}/(1-rho1^nn.AdamTime);
            newR = nn.rb{k}/(1-rho2^nn.AdamTime);
            nn.b{k} = nn.b{k} - nn.learning_rate*newS./sqrt(newR+0.00001);  %rho1 = 0.9, rho2 = 0.999, delta = 0.00001
        end;
    else
        if strcmp(nn.optimization_method, 'normal')
            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k};
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k};
            nn.Gamma{k} = nn.Gamma{k} - nn.learning_rate*nn.Gamma_grad{k};
            nn.Beta{k}= nn.Beta{k} - nn.learning_rate*nn.Beta_grad{k};
        elseif strcmp(nn.optimization_method, 'AdaGrad')
            nn.rW{k} = nn.rW{k} + nn.W_grad{k}.^2;
            nn.rb{k} = nn.rb{k} + nn.b_grad{k}.^2;
            nn.rGamma{k} = nn.rGamma{k} +  nn.Gamma_grad{k}^2;
            nn.rBeta{k} = nn.rBeta{k} +  nn.Beta_grad{k}^2;
            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k}./(sqrt(nn.rW{k})+0.001);
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k}./(sqrt(nn.rb{k})+0.001);
            nn.Gamma{k} = nn.Gamma{k} - nn.learning_rate*nn.Gamma_grad{k}/(sqrt(nn.rGamma{k})+0.001);
            nn.Beta{k}= nn.Beta{k} - nn.learning_rate*nn.Beta_grad{k}/(sqrt(nn.rBeta{k})+0.001);
        elseif strcmp(nn.optimization_method, 'RMSProp')
            nn.rW{k} = 0.9*nn.rW{k} + 0.1*nn.W_grad{k}.^2;
            nn.rb{k} = 0.9*nn.rb{k} + 0.1*nn.b_grad{k}.^2;
            nn.rGamma{k} = 0.9*nn.rGamma{k} +  0.1*nn.Gamma_grad{k}^2;
            nn.rBeta{k} = 0.9*nn.rBeta{k} +  0.1*nn.Beta_grad{k}^2;
            nn.W{k} = nn.W{k} - nn.learning_rate*nn.W_grad{k}./(sqrt(nn.rW{k})+0.001);
            nn.b{k} = nn.b{k} - nn.learning_rate*nn.b_grad{k}./(sqrt(nn.rb{k})+0.001);
            nn.Gamma{k} = nn.Gamma{k} - nn.learning_rate*nn.Gamma_grad{k}/(sqrt(nn.rGamma{k})+0.001);
            nn.Beta{k}= nn.Beta{k} - nn.learning_rate*nn.Beta_grad{k}/(sqrt(nn.rBeta{k})+0.001); %rho = 0.9
        elseif strcmp(nn.optimization_method, 'Momentum')
            rho = 0.1;%rho = 0.1;
            nn.vW{k} = rho*nn.vW{k} - nn.learning_rate*nn.W_grad{k};
            nn.vb{k} = rho*nn.vb{k} - nn.learning_rate*nn.b_grad{k};
            nn.vGamma{k} = rho*nn.vGamma{k} - nn.learning_rate*nn.Gamma_grad{k};
            nn.vBeta{k} = rho*nn.vBeta{k} - nn.learning_rate*nn.Beta_grad{k};
            nn.W{k} = nn.W{k} + nn.vW{k};
            nn.b{k} = nn.b{k} + nn.vb{k}; 
            nn.Gamma{k} = nn.Gamma{k} + nn.vGamma{k};
            nn.Beta{k} = nn.Beta{k} + nn.vBeta{k};
            
        elseif strcmp(nn.optimization_method, 'Adam') %rho1=0.9,rho2 =0.999
            rho1=0.9;
            rho2 =0.999;
            nn.sW{k} = rho1*nn.sW{k} + (1-rho1)*nn.W_grad{k};
            nn.sb{k} = rho1*nn.sb{k} + (1-rho1)*nn.b_grad{k};
            nn.sGamma{k} = rho1*nn.sGamma{k} + (1-rho1)*nn.Gamma_grad{k};
            nn.sBeta{k} = rho1*nn.sBeta{k} + (1-rho1)*nn.Beta_grad{k};
            nn.rW{k} = rho2*nn.rW{k} + (1-rho2)*nn.W_grad{k}.^2;
            nn.rb{k} = rho2*nn.rb{k} + (1-rho2)*nn.b_grad{k}.^2;
            nn.rBeta{k} = rho2*nn.rBeta{k} + (1-rho2)*nn.Beta_grad{k}.^2;
            nn.rGamma{k} = rho2*nn.rGamma{k} + (1-rho2)*nn.Gamma_grad{k}.^2;   
            
            newS = nn.sW{k}/(1-rho1^nn.AdamTime);
            newR =  nn.rW{k}/(1-rho2^nn.AdamTime);
            nn.W{k} = nn.W{k} - nn.learning_rate*newS./sqrt(newR+0.00001);
            
            newS = nn.sb{k}/(1-rho1^nn.AdamTime);
            newR =  nn.rb{k}/(1-rho2^nn.AdamTime);
            nn.b{k} = nn.b{k} - nn.learning_rate*newS./sqrt(newR+0.00001); 
            
            newS = nn.sGamma{k}/(1-rho1^nn.AdamTime);
            newR =  nn.rGamma{k}/(1-rho2^nn.AdamTime);
            nn.Gamma{k} = nn.Gamma{k} - nn.learning_rate*newS./sqrt(newR+0.00001);
            
            newS = nn.sBeta{k}/(1-rho1^nn.AdamTime);
            newR =  nn.rBeta{k}/(1-rho2^nn.AdamTime);
            nn.Beta{k} = nn.Beta{k} - nn.learning_rate*newS./sqrt(newR+0.00001);%rho1 = 0.9, rho2 = 0.999, delta = 0.00001
        end;

    end
end