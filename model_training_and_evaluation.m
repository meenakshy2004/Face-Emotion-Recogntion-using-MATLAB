% Face Emotion Recognition - Model Training and Evaluation
% This script implements facial emotion recognition using two approaches:
% 1) Transfer Learning with fine-tuning of AlexNet
% 2) Feature Extraction using AlexNet + SVM classifier
% The script uses preprocessed RGB images resized to 227×227.


clc;
clear all;
close all;

%% Transfer learning Approach

imds = imageDatastore('ProcessedData2', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

net = alexnet;

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%% Network Training
%Uncomment the section if training is required.

options = trainingOptions('sgdm',...
     'MiniBatchSize',5,...
     'MaxEpochs',10,...
     'InitialLearnRate',0.0001);
 

%% Fine-tune the network using |trainNetwork| on the new layer array.
netTransfer = trainNetwork(imdsTrain,layers,options);
save('netTransfer.mat', 'netTransfer');

load('netTransfer');
predictedLabels = classify(netTransfer,imdsValidation);

%%
idx = [1 33 20 41];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    
    I = readimage(imdsValidation,idx(i));
    label = predictedLabels(idx(i));
    
    imshow(I)
    title(char(label))
    drawnow
end

%% Computing Classification Accuracy (Transfer Learning)

testLabels = imdsValidation.Labels;
accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels)

%% Feature Extraction Approach (AlexNet + SVM)

net = alexnet;

layer = 'fc7';
featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,imdsValidation,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsValidation.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);

idx = [1 5 10 15];

figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

%% Compute Classification Accuracy (Feature Extraction)

accuracy = mean(YPred == YTest)
save('main.mat');
