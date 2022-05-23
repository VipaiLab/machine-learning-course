function [wrongs,success_ratio,nn] = nn_test(nn,test_x,test_y)
    nn = nn_predict(nn,test_x);
    y_output = nn.a{nn.depth};
    y_output = y_output';
    [~,label] = max(y_output,[],2); %max(a,[],dim)dim=2时比较矩阵a的行,label给出行最大值所在下标
    [~,expection] = max(test_y,[],2);%打开test_y就能理解
    wrongs = find(label ~= expection);
    success_ratio = 1-numel(wrongs)/size(test_y,1);
end
    