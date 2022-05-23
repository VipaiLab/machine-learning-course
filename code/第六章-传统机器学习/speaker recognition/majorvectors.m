function [newVQ,vectornumber]=majorvectors(cepstrum,n)
M=convmatrix(cepstrum);
p=size(cepstrum);
numcoeff=p(1);
numvec=p(2);
for i=1:numcoeff
    cepstrum(i,:)=cepstrum(i,:)*sqrt(M(i));
end;
inc=floor((numvec-1)/(n-1));
newVQ=[];
for i=0:n-1
    newVQ=[newVQ,cepstrum(:,1+i*inc)];
end;
vectornumber=zeros(1,numvec);
difference=inf;

while (difference>0.01*numcoeff*n)
    oldVQ=newVQ;
    for i=1:numvec
        mindistance=inf;
        for j=1:n
            distance=0;
            for k=1:numcoeff
                distance=distance+(cepstrum(k,i)-oldVQ(k,j))^2;
            end;
            if distance<mindistance
                mindistance=distance;
                vectornumber(i)=j;
            end;
        end;
    end;
    newVQ=zeros(numcoeff,n);
    p=zeros(1,n);
    for i=1:numvec
        newVQ(:,vectornumber(i))=newVQ(:,vectornumber(i))+cepstrum(:,i);
        p(vectornumber(i))=p(vectornumber(i))+1;
    end;
    i=0;
    while i<length(p)
        i=i+1;
        if p(i)==0
            p(i)=[];
            newVQ(:,i)=[];
            i=i-1;
        end;
    end;
    n=length(p);
    
    for i=1:n
        newVQ(:,i)=newVQ(:,i)/p(i);
    end;
    difference=0;
    for i=1:n
        for j=1:numcoeff
            difference=difference+(oldVQ(j,i)-newVQ(j,i))^2;
        end;
    end;
end;

        
            
        

