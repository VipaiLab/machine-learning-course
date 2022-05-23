function result=removesilence(y,fs,windowtime)
upperthresholdparameter=0.5;
lowerthresholdparameter=0.3;
zerocrossingparameter=1;
maxframecount=25;
noiseenergy=0;
noisezerocrossing=0;

for i=1:(length(y)-1);
    noiseenergy=noiseenergy+y(i)^2;
    if y(i)*y(i+1)<0
        noisezerocrossing=noisezerocrossing+1;
    end;
end;
noiseenergy=noiseenergy/i;
noisezerocrossing=noisezerocrossing*windowtime*fs/(1000*length(y));
upperthreshold=upperthresholdparameter*noiseenergy;
lowerthreshold=lowerthresholdparameter*noiseenergy;
pointsperframe=round(windowtime*fs/1000);
framenumber=floor(length(y)/pointsperframe);
frameenergy=zeros(framenumber,1);
framezerocrossing=zeros(framenumber,1);
for i=1:framenumber
    startpoint=1+(i-1)*pointsperframe;
    endpoint=i*pointsperframe;
    for j=startpoint:endpoint-1
        frameenergy(i)=frameenergy(i)+y(j)^2;
        if y(j)*y(j+1)<0
            framezerocrossing(i)=framezerocrossing(i)+1;
        end;
    end;
    frameenergy(i)=frameenergy(i)/pointsperframe;
end;

lowerbound=1;
i=1;

result=[];
while(lowerbound<framenumber)
    while(i<framenumber && frameenergy(i)<upperthreshold)
        i=i+1;
    end;
    j=i;
    while(j<framenumber && frameenergy(j)>upperthreshold)
        j=j+1;
    end;
    while(i<=framenumber && i>lowerbound && frameenergy(i)>lowerthreshold)
        i=i-1;
    end;
    while(j<framenumber && frameenergy(j)>lowerthreshold)
        j=j+1;
    end;
    u=i;
    while(i<=framenumber && i>lowerbound && framezerocrossing(i)>zerocrossingparameter*noisezerocrossing && u-i<maxframecount)
        i=i-1;
    end;
    u=j;
    while(j<framenumber && framezerocrossing(j)>zerocrossingparameter*noisezerocrossing &&j-u<maxframecount)
        j=j+1;
    end;
    startpoint=1+(i-1)*pointsperframe;
    stoppoint=j*pointsperframe;
    result=[result,y(startpoint:stoppoint)'];
    lowerbound=j+1;
    i=lowerbound;
end;
result=result';     