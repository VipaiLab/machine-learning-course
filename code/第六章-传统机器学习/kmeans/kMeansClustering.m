A = imread('Lenna.png');
C = makecform('srgb2lab');
ALab = applycform(A,C);
[M,N,P] = size(ALab);
labMetrics = zeros(3,M*N);
flag = 0;
for i=1:M
    for j=1:N
        flag = flag+1;
        labMetrics(1, flag) = ALab(i,j,1);
        labMetrics(2, flag) = ALab(i,j,2);
        labMetrics(3, flag) = ALab(i,j,3);
    end;
end;

K =16;
index = randperm(M*N);
index = index(1:K);
R = zeros(1,M*N);
originalTotalDifference = 0;
for i = 1:M*N
    a = labMetrics(:,i);
    minValue = inf;
    for j = 1:K
        b = labMetrics(:,index(j));
        difference = deltaE2000(a',b');
        if difference <=minValue
            minValue = difference;
            minIndex = j;
        end;
    end;
    R(i) = minIndex;
    originalTotalDifference = originalTotalDifference + minValue;
end;

flag = 1;
delta = 0.001;
while flag
    means = zeros(3,K);
    num = zeros(1,K);
    for i = 1:M*N
        index = R(i);
        means(:,index) = means(:,index) + labMetrics(:,i);
        num(index) = num(index) +1;
    end;
    for i=1:K
        means(:,i) = means(:,i)/num(i);
    end;
    totalDifference = 0;
    for i = 1:M*N
        a = labMetrics(:,i);
        minValue = inf;
        for j = 1:K
            b = means(:,j);
            difference = deltaE2000(a',b');
            if difference <=minValue
                minValue = difference;
                minIndex = j;
            end;
        end;
        R(i) = minIndex;
        totalDifference = totalDifference +minValue;
    end;
    if originalTotalDifference-totalDifference<delta*M*N
        flag = 0;
    else
        originalTotalDifference = totalDifference;
    end;
end;

for i=1:M*N
    index = R(i);
    labMetrics(:,i) = means(:,index);
end;

flag = 0;
for i=1:M
    for j = 1:N
        flag = flag+1;
        ALab(i,j,1) = labMetrics(1,flag);
        ALab(i,j,2) = labMetrics(2,flag);
        ALab(i,j,3) = labMetrics(3,flag);
    end;
end;
C = makecform('lab2srgb');
newA = applycform(ALab,C);
imwrite(uint8(newA), 'lennaProcessedK64.bmp','bmp');

        

    