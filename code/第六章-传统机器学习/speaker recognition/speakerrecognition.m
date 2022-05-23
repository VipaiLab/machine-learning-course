%****************************************************
clear all;
DIR='database';
directory=dir(DIR);
directory(1:2)=[];

%%%%%%%%%%%%%%%%%Feature Extraction%%%%%%%%%%%%%%%%%%
speaker = struct('name',[],'files',[]);
flag = 0;
for i = 256:length(directory) 
    i
    if directory(i).isdir
        flag = flag+1;
        speaker(flag).name = directory(i).name;
        subDir = dir([DIR,'\',speaker(flag).name]);
        subDir(1:2) = [];
        speaker(flag).files = struct('name',[],'features',[]);
        flagFiles = 0;
        for j = 1:length(subDir)
            if strcmp(subDir(j).name(end-2:end),'wav')
                flagFiles = flagFiles+1;
                speaker(flag).files(flagFiles).name = subDir(j).name;
                
                 [y,fs]=audioread([DIR,'\',directory(i).name,'\',subDir(j).name]);
                 y=removesilence(y,fs,10);
                 speaker(flag).files(flagFiles).features = melcepst(y,fs,'dD',12,floor(3*log(fs)),fs/50)';
            end;
        end;
    end;
end;
save speaker.mat speaker;

%%%%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%%%%
load speaker.mat;
DIR='database';    
M=32;
mindelta=0.01;
mindifference=0.0001;
%THE RATIO OF FILES FOR TRAINING;
trainingRatio = 0.5;
for i = 1:length(speaker)
    i
    cepstrum = [];
    for j = 1:floor(length(speaker(i).files)*trainingRatio)
        cepstrum = [cepstrum,speaker(i).files(j).features];
    end;
    [speaker(i).probability,speaker(i).average,speaker(i).delta]=trainGMM(cepstrum,M,mindelta,mindifference);
end;
save speaker.mat speaker;    

%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%
load speaker.mat;
totalFileNumber = 0;
correctFileNumber = 0;
for i = 1:length(speaker)
    i
    for j = floor(length(speaker(i).files)*trainingRatio)+1:length(speaker(i).files)
        totalFileNumber = totalFileNumber + 1;
        score = zeros(1,length(speaker));
        for k = 1:length(speaker)
            score(k) = distanceGMM(speaker(i).files(j).features,speaker(k).probability,speaker(k).average,speaker(k).delta);
        end;
        [maxScore,index] = max(score);
        if i == index
            correctFileNumber = correctFileNumber +1;
        end;
    end;
end;
recognitionRate = correctFileNumber/totalFileNumber;

