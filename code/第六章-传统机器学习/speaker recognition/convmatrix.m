function M=convmatrix(cepstrum)
p=size(cepstrum);
numcoeff=p(1);
numvec=p(2);
M=zeros(numcoeff,1);
for i=1:numvec
    U=cepstrum(:,i).^2;
    M=M+U;
end;
for i=1:numcoeff
    M(i)=numvec/M(i);
end;

    