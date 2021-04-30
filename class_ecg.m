clc;
clear
close all
warning off;
addpath('support');
C=pwd;
%unzip(fullfile(D,'physionet_ECG_data-master.zip'),D)
load('ECGData.mat')  
% Explanation :
% this file consists of 162 files 
% label ARR-Arrhythmia, CHF-Congestive Heart Failure, NSR-Normal Sinus Rhythm
%The first 96 rows  of 'ARR' are modified copies of the two ECG recordings in the 48 data files contained in the MIT-BIH Arrhythmia Database.
%The next 30 rows of 'CHF' are modified copies of the two ECG recordings in the 15 data files contained in The BIDMC Congestive Heart Failure Database.
%The final 36 rows of 'NSR' are modified copies of the two ECG recordings in the 18 data files contained in the MIT-BIH Normal Sinus Rhythm Database.

 parentDir = tempdir;
dataDir = 'data';
 helperCreateECGDirectories(ECGData,parentDir,dataDir)
 helperPlotReps(ECGData)

Fs = 128;
fb = cwtfilterbank('SignalLength',1000,...
    'SamplingFrequency',Fs,...
    'VoicesPerOctave',12);
sig = ECGData.Data(1,1:1000);
[cfs,frq] = wt(fb,sig);
t = (0:999)/Fs;figure;pcolor(t,frq,abs(cfs));
set(gca,'yscale','log');shading interp;axis tight;
title('Scalogram');xlabel('Time (s)');ylabel('Frequency (Hz)')

helperCreateRGBfromTF(ECGData,parentDir,dataDir)% uncommment this when
% %directory creation is required 
% 
 allImages = imageDatastore(fullfile(parentDir,dataDir),'IncludeSubfolders',true,'LabelSource','foldernames');

rng default
[Tra,test]=splitEachLabel(allImages,0.8,'randomized');
disp(['Number of training signals: ',num2str(numel(Tra.Files))]);
disp(['Number of validation signals: ',num2str(numel(test.Files))]);

net = googlenet;
lgraph = layerGraph(net);
numberOfLayers = numel(lgraph.Layers);
lgraph = removeLayers(lgraph,{'pool5-drop_7x7_s1','loss3-classifier','prob','output'});

numClasses = numel(categories(Tra.Labels));
newLayers = [
    dropoutLayer(0.6,'Name','newDropout')
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-7x7_s1','newDropout');
inputSize = net.Layers(1).InputSize;
options = trainingOptions('sgdm',...
    'MiniBatchSize',15,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'ValidationData',test,...
    'ValidationFrequency',10,...
    'ValidationPatience',Inf,...
    'Verbose',1,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');
rng default
trainednwk = trainNetwork(Tra,lgraph,options);
save trainednwk trainednwk
load trainednwk
trainednwk.Layers(end-2:end)
cNames=trainednwk.Layers(end).ClassNames

[YPred,probs]=classify(trainednwk,test);
accuracy = mean(YPred==test.Labels)
