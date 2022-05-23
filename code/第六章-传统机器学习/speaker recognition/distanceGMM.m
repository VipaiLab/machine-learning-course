function dist=distanceGMM(cepstrum,probability,average,delta)
% This function calculate the probability that a testing data belongs to a Gaussian
% mixture model. 
% Input: 
%     cepstrum: The cepstrum of a testing data, which is a U*V matrix. 
%               U is the dimension of the cepstrum  vectors, and V is the
%               number of cepstrum vectors.
%     probability, average, delta are the model parameters of the GMM.
% Output: 
%     dist: the logarithm of the probability that the testing data belongs
%     to a Gaussian mixture model.
p=size(cepstrum);
vectordimension=p(1);
vectornumber=p(2);
p=size(average);
M=p(2);
dist=0;
product=log(delta);
product=-0.5*sum(product);
for j=1:vectornumber
    p=0;
    for i=1:M
        expfactor=((cepstrum(:,j)-average(:,i)).^2)./delta(:,i);  
        variable=-0.5*sum(expfactor)+product(i);
        p=p+probability(i)*exp(variable);
    end;
    dist=dist+log(p);
end;
dist=dist/vectornumber;
