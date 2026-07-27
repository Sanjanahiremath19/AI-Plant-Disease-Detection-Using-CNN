clc; clear; close all;
load('C:\Users\Vaishnavi Kamat\OneDrive\Desktop\plant_disease\plant_model.mat', 'trainedNet');
net = trainedNet;

[filename, pathname] = uigetfile({'*.jpg;*.png'}, 'Select Leaf Image');
if isequal(filename,0), return; end

img = imread(fullfile(pathname, filename));
img_resized = imresize(img, [224 224]);

[label, scores] = classify(net, img_resized);
confidence = max(scores);

label_str = strrep(char(label), '_', ' ');
figure('Name','Result','NumberTitle','off');
imshow(img);
title(sprintf('%s (%.1f%% Confidence)', label_str, confidence*100));
