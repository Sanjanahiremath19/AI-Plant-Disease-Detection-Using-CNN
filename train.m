clc;
clear;
close all;

%% LOAD DATASET
datasetPath = 'C:\Users\Vaishnavi Kamat\OneDrive\Desktop\plant_disease\dataset';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% SPLIT DATA
[trainDS, testDS] = splitEachLabel(imds, 0.8, 'randomized');

%% LOAD MOBILENETV2
net = mobilenetv2;

inputSize = net.Layers(1).InputSize;

%% DATA AUGMENTATION
augmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandScale', [0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize(1:2), trainDS, ...
    'DataAugmentation', augmenter);

augTest = augmentedImageDatastore(inputSize(1:2), testDS);

%% MODIFY NETWORK
lgraph = layerGraph(net);

numClasses = numel(categories(trainDS.Labels));

newFCLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph,'Logits',newFCLayer);
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',newClassLayer);

%% TRAINING OPTIONS
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 3, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augTest, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% TRAIN NETWORK
trainedNet = trainNetwork(augTrain, lgraph, options);

%% TEST ACCURACY
predLabels = classify(trainedNet, augTest);
accuracy = mean(predLabels == testDS.Labels);

fprintf('Accuracy = %.2f%%\n', accuracy * 100);

%% SAVE MODEL
save('C:\Users\Vaishnavi Kamat\OneDrive\Desktop\plant_disease\plant_model.mat', 'trainedNet');

disp('MobileNetV2 model trained successfully!');
