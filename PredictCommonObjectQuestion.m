%% Code to predict if a question in the VQA dataset corresponds to a common object (person, animal or food) or not
%% MATLAB R2015a or b recommended
%% Author: Nitin J. Sanket, MSE in Robotics Student, University of Pennsylvania, nitinsan@seas.upenn.edu
clc
clear all
close all

% Add package paths
addpath('./liblinear');
addpath('./libsvm');

%% Read the Text File as Questions
% Run vqaGetQuestions.py to get Questions.txt 
fid = fopen('Questions.txt'); 
count = 1;

tline = fgets(fid);
Questions{1} = tline;
while ischar(tline)
    tline = fgets(fid);
    Questions{count} = tline;
    count=count+1;
end

fclose(fid);

Questions(count-1) = [];

%% Read the Text File as Words and process them
Words = textread('Questions.txt','%s');

for i = 1:size(Words,1)
    Words{i} = lower(Words{i});
    Words{i}(regexp(Words{i},'[!?,.#@&"-_]'))=[];
    Words{i} = strrep(Words{i}, '''"', '');
    WordsStemmed{i} = porterStemmer(Words{i});
end

% Find unique stemmed words
[~,uniqueIndices] = unique(WordsStemmed);
for i = 1:size(uniqueIndices,1)
    WordsUniqueStemmed{i} = WordsStemmed{uniqueIndices(i)};
end

%% Count the words now (Make Bag of Words Model)
RawCounts = zeros(length(Questions),length(WordsUniqueStemmed));
for i = 1:length(Questions)
    WordsNow =  lower(strsplit(Questions{i},' '));
    for j = 1:length(WordsNow)
        WordsNow{j}(regexp(WordsNow{j},'[!?,.#@&"-_]'))=[];
        WordsNow{j} = strrep(WordsNow{j}, '''"', '');
        WordNowStemmed = porterStemmer(WordsNow{j});
        WordNowStemmed = strtrim(WordNowStemmed);
        WordIdx = strmatch(WordNowStemmed, WordsUniqueStemmed, 'exact');
        RawCounts(i,WordIdx) = RawCounts(i,WordIdx)+1;
    end
end


% labels = zeros(length(Questions),1);
%% Label the questions
% % Left click is 1 and Right click is 0
% close all
% figure('units','normalized','outerposition',[0 0 1 1]);
% for iter = 901:length(Questions)
%     disp(iter);
%     imshow(zeros(100,100));
%     title([num2str(iter),' : ',Questions{iter}]);
%     [~,~,button] = ginput(1);
%     if button == 1
%         labels(iter) = 2;
%     else
%         labels(iter) = 0;
%     end
% end
% load('labelsAnimal.mat');

%% Pick Test and Train Sets
% TrainIdxs = randperm(length(labels),600); % 600 Train
% TestIdxs = setdiff(1:length(labels), TrainIdxs); % Remaining Test
% load('TestTrainIdxs600.mat');

% labelsA = zeros(length(Questions),1);
% labelsF = zeros(length(Questions),1);
% labelsP = zeros(length(Questions),1);
% labelsA(AnimalIdxs) = 1;
% labelsF(FoodIdxs) = 1;
% labelsP(PersonIdxs) = 1;
% 
% % Stratified CVPartition
% cvPartitionA = cvpartition(labelsA,'Holdout',.40);
% TrainIdxsA = training(cvPartitionA,1);
% TestIdxsA = test(cvPartitionA,1);
% cvPartitionF = cvpartition(labelsF,'Holdout',.40);
% TrainIdxsF = training(cvPartitionF,1);
% TestIdxsF = test(cvPartitionF,1);
% cvPartitionP = cvpartition(labelsP,'Holdout',.40);
% TrainIdxsP = training(cvPartitionP,1);
% TestIdxsP = test(cvPartitionP,1);

% cvPartitionAFP = cvpartition(labelsP,'Holdout',.40);
% TrainIdxsAFP = training(cvPartitionAFP,1);
% TestIdxsAFP = test(cvPartitionAFP,1);


% save('TestTrain600StratAFP.mat');
load('TestTrain600StratAFP.mat');
load('TestTrainIdxs600AFPAll.mat');

%% Rank Features
NumFeat = length(TrainCounts);
SelIdxs = (1:NumFeat); % Use all features
labelsP = double(labelsP | labelsA | labelsF);

TrainCounts = RawCounts(TrainIdxsAFP, SelIdxs);
TestCounts = RawCounts(TestIdxsAFP, SelIdxs);
TrainLabels = labelsP(TrainIdxsAFP);
TestLabels = labelsP(TestIdxsAFP);

% [RankedIdxs, ~] = rankfeatures(TrainCounts', TrainLabels); % Ranked Features not used

%% Fit a model and measure accuracy
SVMModel = fitcsvm(TrainCounts,TrainLabels);
TrainPred1 = predict(SVMModel, TrainCounts);
TestPred1 = predict(SVMModel, TestCounts);
disp(['Train Accuracy SVM Linear ', num2str(sum(TrainPred1==TrainLabels)./length(TrainLabels).*100)]);
disp(['Test Accuracy SVM Linear ', num2str(sum(TestPred1==TestLabels)./length(TestLabels).*100)]);


%% LogitBoost
XTrain =  TrainCounts;
YTrain = TrainLabels;
XTest =  TestCounts;
YTest = TestLabels;
ABModel = fitensemble(XTrain,YTrain,'LogitBoost',1000,'Tree','NPrint',100);
[TrainPred2, ~] = predict(ABModel, XTrain);
[TestPred2, ~] = predict(ABModel, XTest);
disp(['Train Accuracy LB ', num2str(sum(TrainPred2==TrainLabels)./length(TrainLabels).*100)]);
disp(['Test Accuracy LB ', num2str(sum(TestPred2==TestLabels)./length(TestLabels).*100)]);

%% RobustBoost
XTrain =  TrainCounts;
YTrain = TrainLabels;
XTest =  TestCounts;
YTest = TestLabels;
ABModel = fitensemble(XTrain,YTrain,'RobustBoost',5000,'Tree','NPrint',100);
[TrainPred3, ~] = predict(ABModel, XTrain);
[TestPred3, ~] = predict(ABModel, XTest);
disp(['Train Accuracy RB ', num2str(sum(TrainPred3==TrainLabels)./length(TrainLabels).*100)]);
disp(['Test Accuracy RB ', num2str(sum(TestPred3==TestLabels)./length(TestLabels).*100)]);

%% Logisitic Regression 
XTrain =  TrainCounts;
YTrain = TrainLabels;
XTest =  TestCounts;
YTest = TestLabels;
LRArgs = 'col -b 1 -C -s 0';
LRModel = train(YTrain, sparse(XTrain), LRArgs);
[TrainPred4, ~, ~] = predict(YTrain, sparse(XTrain), LRModel, ['-q', 'col', '-b 1']);
[TestPred4, ~, ~] = predict(YTest, sparse(XTest), LRModel, ['-q', 'col', '-b 1']);

disp(['Train Accuracy LR ', num2str(sum(TrainPred4==TrainLabels)./length(TrainLabels).*100)]);
disp(['Test Accuracy LR ', num2str(sum(TestPred4==TestLabels)./length(TestLabels).*100)]);

%% SVM-Intersection Kernel
XTrain =  TrainCounts;
YTrain = TrainLabels;
XTest =  TestCounts;
YTest = TestLabels;
kernel_linear =  @(x,x2) kernel_intersection(x, x2);
[TrainPred5, TestPred5, Model5] = kernel_libsvm(XTrain, YTrain, XTest, YTest, kernel_linear);
disp(['Train Accuracy SVM Intersection ', num2str(sum(TrainPred5==TrainLabels)./length(TrainLabels).*100)]);
disp(['Test Accuracy SVM  Intersection ', num2str(sum(TestPred5==TestLabels)./length(TestLabels).*100)]);

