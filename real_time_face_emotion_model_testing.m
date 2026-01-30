% Real-Time Facial Emotion Recognition using Webcam

% This script performs real-time facial emotion recognition using a webcam.
% It supports two inference approaches:
% 1) Transfer Learning using a fine-tuned AlexNet model
% 2) Feature Extraction using AlexNet deep features with an SVM classifier

% The input frames are resized to 227×227 to match AlexNet requirements.
% Enable ONLY ONE approach at a time by loading the corresponding model.

clc;
clear all ;
close all ;

%% Initialize Webcam 
cam = webcam; 

%% Load Pretrained Model (Choose either Transfer Learning or Feature Extraction Approach) 
% For Transfer Learning Approach
load('netTransfer.mat');  

% or 

% For Feature Extraction Approach
%load('mainmma.mat'); 

%% Real-Time Emotion Detection Loop

while true 
    img = snapshot(cam); 
    img = imresize(img, [227 227]); 

    % Emotion Classification
    
    %Transfer Learning Approach
    predictedLabel = classify(netTransfer, img); 

    % or For Feature Extraction Approach 
    %features = activations(net, img, 'fc7', 'OutputAs', 'rows'); 
    %predictedLabel = predict(classifier, features); 

    % Display Result 
    imshow(img); 
    title(['Predicted Emotion: ', char(predictedLabel)], 'FontSize', 14); 
    drawnow; 
end 

%% Release Webcam Resource

clear cam; 
