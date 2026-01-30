% Pre-processing of the Dataset for Face Emotion Recognition
% This script reads all images from the training dataset, ensures each image
% has three channels (RGB), resizes them to 227×227 (AlexNet input size),
% and saves the processed images into a new directory while preserving
% the original class-wise folder structure.

clc;
clear all;
close all;

imds = imageDatastore('train', ... 
'IncludeSubfolders',true, ... 
'LabelSource','foldernames'); 

numImages = numel(imds.Files); 

rootDir = 'ProcessedData'; 
mkdir(rootDir) 

totFolders = numel(categories(imds.Labels)); 
foldersNames = categories(imds.Labels); 

for n = 1:totFolders 
    mkdir(fullfile(rootDir, foldersNames{n})); 
end 

count = 1; 

for i = 1:numImages 
    I = readimage(imds, i); 

    if size(I, 3) ~= 3 
       I = cat(3, I, I, I); 
    end 
    
    I = imresize(I, [227, 227]); 
    subF = char(imds.Labels(i)); 
    imageName = [sprintf('%d', count), '.tiff']; 
    fullFileName = fullfile(rootDir, subF, imageName); 
    imwrite(I, fullFileName); 
    fprintf('Processed %0.2f percent records\n', (i / numImages) * 100); 
    count = count + 1; 
end 
