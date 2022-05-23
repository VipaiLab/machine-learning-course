function [newprobability,newaverage,newdelta]=trainGMM(cepstrum,M,mindelta,mindifference)
% This function aims at training the Gaussian mixture model. Refer to the following paper for details. 
% 
% Reynolds, D. A. and R. C. Rose (1995). Robust text-independent speaker
% identification using Gaussian mixture speaker models. IEEE Transactions
% on Speech and Audio Processing 3(1), 72-83. 
%
% Input:
%   cepstrum: the input cepstrum, which is a U*V matrix. 
%             U is the dimension of the cepstrum  vectors, and V is the number of cepstrum vectors.   
%   M: the number of Gaussian mixtures
%   mindelta: the minimal value of delta, setting this parameter to prevent the training process ill-posed. 
%   mindifference: this parameter controlls the convergence process of the EM algorithm. 
%
%   Output: newprobability, newaverage, newdelta are the model parameters. 

p=size(cepstrum);
vectordimension=p(1);
vectornumber=p(2);
[oldaverage,vectorgroup]=majorvectors(cepstrum,M);
p=size(oldaverage);
M=p(2);
olddelta=zeros(vectordimension,M);
oldprobability=zeros(M,1);

group=struct('elements',[]);
for i=1:M
    group(i).elements=[];
end;
for i=1:length(vectorgroup)
    group(vectorgroup(i)).elements=[group(vectorgroup(i)).elements,i];
end;
flag=0;
for i=1:M
    for j=1:length(group(i).elements)
        olddelta(:,i)=olddelta(:,i)+(cepstrum(:,group(i).elements(j))-oldaverage(:,i)).^2;
    end;
end;

for i=1:M
    oldprobability(i)=length(group(i).elements);
end;

for i=1:M
    olddelta(:,i)=olddelta(:,i)/oldprobability(i);
end;
olddelta=max(olddelta,mindelta);
oldprobability=oldprobability/vectornumber;

oldaverageprobability=0;
product=ones(1,M);
for i=1:M
    for k=1:vectordimension
        product(i)=product(i)*olddelta(k,i);
    end;
end;
product=sqrt(product);

expfactor=zeros(M,vectornumber);
for j=1:vectornumber
    for i=1:M
        expfactor(i,j)=sum((cepstrum(:,j)-oldaverage(:,i)).^2./olddelta(:,i));
    end;
end;

p=zeros(1,vectornumber);
postprobability=zeros(M,vectornumber);
for i=1:M
    p=p+oldprobability(i)*exp(-0.5*expfactor(i,:))/product(i);
    postprobability(i,:)=oldprobability(i)*exp(-0.5*expfactor(i,:))/product(i);
end;
oldaverageprobability=sum(log(p))/vectornumber;

for i=1:vectornumber
    total=sum(postprobability(:,i));
    if total==0
        postprobability(:,i)=ones(M,1)/M;
    else
        postprobability(:,i)=postprobability(:,i)/total;
    end;
end;  
count=0;
flag=1;
while (flag)
    newprobability=zeros(M,1);
    newaverage=zeros(vectordimension,M);
    newdelta=zeros(vectordimension,M);
   
    for i=1:M
        newprobability(i)=sum(postprobability(i,:));
        newaverage(:,i)=newaverage(:,i)+cepstrum*postprobability(i,:)';
        newdelta(:,i)=newdelta(:,i)+cepstrum.^2*postprobability(i,:)';
        newaverage(:,i)=newaverage(:,i)/newprobability(i);
        newdelta(:,i)=newdelta(:,i)/newprobability(i)-newaverage(:,i).^2;
        newprobability(i)=newprobability(i)/vectornumber;
    end;
    newdelta=max(newdelta,mindelta);
    
    newaverageprobability=0;
    product=ones(1,M);
    for i=1:M
        for k=1:vectordimension
            product(i)=product(i)*newdelta(k,i);
        end;
    end;
    product=sqrt(product);

    for j=1:vectornumber
        for i=1:M
            expfactor(i,j)=sum((cepstrum(:,j)-newaverage(:,i)).^2./newdelta(:,i));
        end;
    end;

    p=zeros(1,vectornumber);
    postprobability=zeros(M,vectornumber);
    for i=1:M
        p=p+newprobability(i)*exp(-0.5*expfactor(i,:))/product(i);
        postprobability(i,:)=newprobability(i)*exp(-0.5*expfactor(i,:))/product(i);
    end;
    newaverageprobability=sum(log(p))/vectornumber;
    
    for i=1:vectornumber
        total=sum(postprobability(:,i));
        if total==0
            postprobability(:,i)=ones(M,1)/M;
        else
            postprobability(:,i)=postprobability(:,i)/total;
        end;
    end;  

    if abs(newaverageprobability-oldaverageprobability)<mindifference
        flag=0;
    else
        oldaverageprobability=newaverageprobability;
    end;
end;

% p=size(cepstrum);
% vectordimension=p(1);
% vectornumber=p(2);
% [oldaverage,vectorgroup]=majorvectors(cepstrum,M);
% p=size(oldaverage);
% M=p(2);
% olddelta=zeros(vectordimension,M);
% oldprobability=zeros(M,1);
% 
% group=struct('elements',[]);
% for i=1:M
%     group(i).elements=[];
% end;
% for i=1:length(vectorgroup)
%     group(vectorgroup(i)).elements=[group(vectorgroup(i)).elements,i];
% end;
% 
% for i=1:M
%     for j=1:length(group(i).elements)
%         olddelta(:,i)=olddelta(:,i)+(cepstrum(:,group(i).elements(j))-oldaverage(:,i)).^2;
%     end;
% end;
% 
% for i=1:M
%     oldprobability(i)=length(group(i).elements);
% end;
% 
% for i=1:M
%     olddelta(:,i)=olddelta(:,i)/oldprobability(i);
% end;
% olddelta=max(olddelta,mindelta);
% oldprobability=oldprobability/vectornumber;
% 
% oldaverageprobability=0;
% product=ones(vectordimension,M);
% for i=1:M
%     for k=1:vectordimension
%         product(k,i)=product(k,i)*olddelta(k,i);
%     end;
% end;
% for j=1:vectornumber
%     p=0;
%     for i=1:M
%         product=1;
%         expfactor=0;
%         for k=1:vectordimension
%             product=product*olddelta(k,i);
%             expfactor=expfactor+(cepstrum(k,j)-oldaverage(k,i))^2/olddelta(k,i);
%         end;
%         p=p+oldprobability(i)*exp(-0.5*expfactor)/sqrt(product);
%         postprobability(i,j)=oldprobability(i)*exp(-0.5*expfactor)/sqrt(product);
%     end;
%     oldaverageprobability=oldaverageprobability+log(p);
% end;
% oldaverageprobability=oldaverageprobability/vectornumber;
% 
%  
% for i=1:vectornumber
%     total=0;
%     for j=1:M
%         total=total+postprobability(j,i);
%     end;
%     if total==0
%         postprobability=1/M;
%     else
%         postprobability(:,i)=postprobability(:,i)/total;
%     end;
% end;  
% count=0;
% flag=1;
% while (flag)
%     newprobability=zeros(M,1);
%     newaverage=zeros(vectordimension,M);
%     newdelta=zeros(vectordimension,M);
%     for i=1:M
%         for j=1:vectornumber
%             newprobability(i)=newprobability(i)+postprobability(i,j);
%             newaverage(:,i)=newaverage(:,i)+postprobability(i,j)*cepstrum(:,j);
%             newdelta(:,i)=newdelta(:,i)+postprobability(i,j)*cepstrum(:,j).^2;
%         end;
%         newaverage(:,i)=newaverage(:,i)/newprobability(i);
%         newdelta(:,i)=newdelta(:,i)/newprobability(i)-newaverage(:,i).^2;
%         newprobability(i)=newprobability(i)/vectornumber;
%     end;
%     newdelta=max(newdelta,mindelta);
%     
%     postprobability=zeros(M,vectornumber);
%     newaverageprobability=0;
%     for j=1:vectornumber
%         j
%         p=0;
%         for i=1:M
%             product=1;
%             expfactor=0;
%             for k=1:vectordimension
%                 product=product*newdelta(k,i);
%                 expfactor=expfactor+(cepstrum(k,j)-newaverage(k,i))^2/newdelta(k,i);
%             end;
%             p=p+newprobability(i)*exp(-0.5*expfactor)/sqrt(product);
%             postprobability(i,j)=newprobability(i)*exp(-0.5*expfactor)/sqrt(product);
%         end;
%         newaverageprobability=newaverageprobability+log(p);
%     end;
%     newaverageprobability=newaverageprobability/vectornumber
%     for i=1:vectornumber
%         total=0;
%         for j=1:M
%             total=total+postprobability(j,i);
%         end;
%         if total==0
%             postprobability(:,i)=1/M;
%         else
%             postprobability(:,i)=postprobability(:,i)/total;
%         end;
%     end;       
%     sign=(newaverageprobability-oldaverageprobability);
%     if abs(newaverageprobability-oldaverageprobability)<mindifference
%         flag=0;
%     else
%         oldaverageprobability=newaverageprobability;
%     end;
% end;
