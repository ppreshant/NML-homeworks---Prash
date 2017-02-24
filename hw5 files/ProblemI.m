function ProblemI
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I,
% ProblemIV
% 

foldCount = 3;

trainDataFilename = 'iris-train copy.txt';
testDataFilename = 'iris-test copy.txt';

[trainInput,trainOutput] = loadData(trainDataFilename);
[testInput,testOutput] = loadData(testDataFilename);

folds = crossValidation(foldCount, trainInput, trainOutput, testInput, testOutput);

batchSize = 1;  % set the size of the batch, i.e. number of patterns per batch

numNodes = [4, 2, 3];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightValues(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 0.1;
% alpha = 0.1;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxIterations = 2000;
errorTolerance = 0.10;

for currentFoldIndex = 1:foldCount
    TrainFoldIn = folds{currentFoldIndex,1};
    TestFoldIn  = folds{currentFoldIndex,3};
    
    TrainFoldOut = folds{currentFoldIndex,2};
    TestFoldOut  = folds{currentFoldIndex,4};
    
    [weightMatrices,trainAccHistory,testAccHistory] = train(TrainFoldIn, TrainFoldOut, TestFoldIn, TestFoldOut,  numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);

    actualTrainOutput = test(TrainFoldIn, tanhSlope, numNodes, weightMatrices);
    thresholdedTrainOutput = threshold(actualTrainOutput);

    actualTestOutput = test(TestFoldIn, tanhSlope, numNodes, weightMatrices);
    thresholdedTestOutput = threshold(actualTestOutput);
    
    trainAcc = classificationAccuracy(TrainFoldOut,thresholdedTrainOutput);
    testAcc = classificationAccuracy(TestFoldOut,thresholdedTestOutput);
    disp(['Training accuracy = ',num2str(trainAcc * 100),'%'])
    disp(['Testing accuracy = ',num2str(testAcc * 100),'%'])
    
    % plot for Training/Testing accuracy
    % figure;
    %
    % plot(sort(trainInput,'descend'),sort(actualTrainOutput),'--');
    % hold on
    % plot(trainInput,trainOutput);
    % plot(sort(testInput,'descend'),sort(actualTestOutput),'--');
    % plot(testInput,testOutput);
    %
    % grid on
    % xlabel('x')
    % ylabel('f(x) = 1/x')
    % title('Comparison of actual training and testing accuracy wrt desired output')
    % legend('Training Learnt Function','Training Actual Function','Testing Learnt Function','Testing Actual Function')
    
    % plot Learning History
    figure; plot(trainAccHistory(2,:),trainAccHistory(1,:)); hold on;  plot(testAccHistory(2,:),testAccHistory(1,:));
    
    grid on
    xlabel('Learning Steps')
    ylabel('Classification accuracy % (misclassified/total)')
    title(['Learning History (Testing Fold: ',num2str(currentFoldIndex),')'])
    legend('Training Accuracy','Testing Accuracy')
    
    [traincM, trainAccuracy] = confusionMatrix(TrainFoldOut,thresholdedTrainOutput);
    [testcM, testAccuracy] = confusionMatrix(TestFoldOut,thresholdedTestOutput);
    
    creativePlots(true,currentFoldIndex,TrainFoldOut,thresholdedTrainOutput)
    creativePlots(false,currentFoldIndex,TestFoldOut,thresholdedTestOutput)
end

end


function [weightMatrices,trainAccHistory,testAccHistory] = train(trainInput, trainOutput, testInput, testOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)

trainAccHistory = zeros(2,size(trainOutput,1));
testAccHistory = zeros(2,size(testOutput,1));

eval_interval = maxIterations / 100;
layerOutputs = cell(1,length(numNodes));

nodeDeltas = createNodeValues(numNodes);
nodeErrorGradients = createNodeValues(numNodes);
weightDeltas = createWeightValues(numNodes);

for i = 1:maxIterations
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);
    for j = 1:length(randomizedInput)
        pattern = randomizedInput(j,:);
        desiredOutput = randomizedOutput(j,:);
        
        % forward propagation
        layerOutputs{1} = pattern;
        layerOutputs{1}(end+1) = 1;
        for k = 1:length(numNodes)-1
            layerOutputs{k+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{k} * layerOutputs{k}')';
            if k ~= length(numNodes) - 1
                layerOutputs{k+1}(end) = 1;
            end
        end
                        
        % backward propagation
        for m = length(layerOutputs):-1:2
            currentLayerOutput = layerOutputs{m};
            previousLayerOutput = layerOutputs{m-1};
            for n = 1:length(currentLayerOutput)
                for p = 1:length(previousLayerOutput)
                    if m == length(layerOutputs)
                        nodeDeltas{m-1}(n) = (desiredOutput(n) - currentLayerOutput(n)) .* hyperbolicTangentDerivative(tanhSlope, weightMatrices{m-1}(n,:) * previousLayerOutput')';
                        nodeErrorGradients{m-1}(n) = -1 .* nodeDeltas{m-1}(n) .* previousLayerOutput(p);
                        weightDeltas{m-1}(n,p) = -learningRate .* nodeErrorGradients{m-1}(n);
                    else
                        nodeDeltas{m-1}(n) = computeHiddenNodeDelta(tanhSlope,m-1,n,layerOutputs,nodeDeltas,weightMatrices);
                        nodeErrorGradients{m-1}(n) = computeHiddenNodeErrorGradient(m-1, n, nodeDeltas, weightMatrices);
                        weightDeltas{m-1}(n,p) = learningRate .* nodeDeltas{m-1}(n) .* previousLayerOutput(p);
                    end
                end
            end
        end
        weightMatrices = updateWeights(numNodes, weightMatrices, weightDeltas);
        
%         testOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);
%         RMSE = computeRMSE(desiredOutput,testOutput);
%         if RMSE < errorTolerance
%             disp('DONE!')
%             return
%         end
    end
    if mod(i,eval_interval) == 0
        dum = i/eval_interval + 1;
        frozenTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);
        frozenTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices);
        thresholdedTrainOutput = threshold(frozenTrainOutput);
        thresholdedTestOutput = threshold(frozenTestOutput);
        trainAcc = classificationAccuracy(trainOutput,thresholdedTrainOutput);
        testAcc = classificationAccuracy(testOutput,thresholdedTestOutput);
        trainAccHistory(1,dum) = trainAcc; trainAccHistory(2,dum) = i*j; % store errors and learning steps
        testAccHistory(1,dum) = testAcc; testAccHistory(2,dum) = i*j;  % store errors and learning steps
        
%         if RMSE_train < errorTolerance
%             total_steps = (i-1)* size(trainInput,1) + k; % steps taken to complete the training
%             Erms_train = Erms_train(:,Erms_train(1,:) ~= 0); Erms_test = Erms_test(:,Erms_test(1,:) ~= 0); % clip the Error storage matrix when terminating
%             otherVariables{1} = total_steps; otherVariables{2} = Erms_train; otherVariables{3} = Erms_test; % giving output variables
%             return
%         end
    end
    
end

end


function testOutput = test(testInput, tanhSlope, numNodes, weightMatrices)

testOutput = zeros(length(testInput),3);

for i = 1:length(testInput)
    output = [testInput(i,:),1]';
    for j = 1:length(numNodes) - 1
        output = hyperbolicTangentFunction(tanhSlope,weightMatrices{j} * output);
        if j ~= length(numNodes) - 1
            output(end) = 1;
        end
    end
    testOutput(i,:) = output;
end

end


function updatedWeights = updateWeights(numNodes, weightMatrices, weightDeltas)

updatedWeights = createWeightValues(numNodes);

for i = 1:length(weightMatrices)
    updatedWeights{i} = weightMatrices{i} + weightDeltas{i};
end

end


function hiddenNodeDelta = computeHiddenNodeDelta(tanhSlope, layerIndex, nodeIndex, layerOutputs, nodeDeltas, weightMatrices)

previousLayerOutput = layerOutputs{layerIndex};
currentLayerWeightVector = weightMatrices{layerIndex}(nodeIndex,:);
derivative = hyperbolicTangentDerivative(tanhSlope, currentLayerWeightVector * previousLayerOutput');

nextLayerWeightVector = weightMatrices{layerIndex+1}(:,nodeIndex);
deltaVector = nodeDeltas{layerIndex+1};
summation = deltaVector * nextLayerWeightVector;

hiddenNodeDelta = derivative .* summation;


end


function hiddenNodeErrorGradient = computeHiddenNodeErrorGradient(layerIndex, nodeIndex, nodeDeltas, weightMatrices)

nextLayerWeightVector = weightMatrices{layerIndex+1}(:,nodeIndex);
deltaVector = nodeDeltas{layerIndex+1};
summation = deltaVector * nextLayerWeightVector;

hiddenNodeErrorGradient = -summation;

end


function weightValues = createWeightValues(numNodes)

numMatrices = length(numNodes) - 1;
weightValues = cell(1, numMatrices);

for i = 1:numMatrices
    weightValues{i} = rand(numNodes(i+1), numNodes(i));
    weightValues{i}(:,end+1) = zeros(1,length(weightValues{i}(:,end)));
    if i ~= numMatrices
        weightValues{i}(end+1,:) = zeros(1,length(weightValues{i}(end,:)));
    end
end

end


function nodeValues = createNodeValues(numNodes)

numLayers = length(numNodes);
nodeValues = cell(1, numLayers - 1);

for i = 2:numLayers
    nodeValues{i-1} = zeros(1,numNodes(i));
end

end


function f = hyperbolicTangentDerivative(a,x)

f = a .* (1 - hyperbolicTangentFunction(a,x) .^ 2);

end


function f = hyperbolicTangentFunction(a,x)

f = (exp(a .* x) - exp(-a .* x)) ./ (exp(a .* x) + exp(-a .* x));

end


function [input,output] = loadData(filename)

fileID = fopen(filename,'r');

formatSpec = '%f %f %f %f';

sizeA = [4, Inf];

A = fscanf(fileID,formatSpec,sizeA)';

fclose(fileID);

input = zeros(size(A,1)/2,size(A,2));
output = zeros(size(A,1)/2,size(A,2)-1);

for i = 1:size(A,1)
    for j = 1:size(A,2)
        if mod(i,2) == 0
            if j ~= 4
                output(i/2,j) = A(i,j);
            end
        else
            input((i+1)/2,j) = A(i,j);
        end
    end
end

end


function thresholdedOutput = threshold(output)

thresholdedOutput = zeros(size(output));

for i = 1:size(output,1)
    for j = 1:size(output,2)
        if output(i,j) == max(output(i,:))
            thresholdedOutput(i,j) = 1;
        end
    end
end

end


function accuracy = classificationAccuracy(desiredOutput, actualOutput)
% input the thresholded actual output
numTrials = size(desiredOutput,1);
numCorrect = 0;

for i = 1:size(desiredOutput,1)
    if sum(desiredOutput(i,:) == actualOutput(i,:)) == 3
        numCorrect = numCorrect + 1;
    end
end

accuracy = numCorrect / numTrials;

end


function creativePlots(train_or_test,i,desiredOutput,thresholdedActualOutput)
% takes the desired and actual outputs and plots the comparision graph
xaxis = 1:size(desiredOutput,1); % create sequence of 1:75 (or number of data points)

[~, desiredSeq] =  max(desiredOutput');  % isolate which category the classification is
[~, actualSeq] =  max(thresholdedActualOutput'); % converts the 2-d matrix into a sequence of numbers

[~, seq] = sort(desiredSeq);    % finds the indices for sorting in ascending order
% oxaxis = xaxis(seq); % for x axis tick values if needed
odesiredSeq = desiredSeq(seq); oactualSeq = actualSeq(seq);

figure;
plot(xaxis,odesiredSeq,'ko'); hold on;
plot(xaxis,oactualSeq,'b.','MarkerSize',15)
ylim([0.5 3.5])
xlim([0 76])
% xlabel('Learning Steps')
yticks([1 2 3])
yticklabels({'Setosa','Versacolor','Virginica'})
ylabel('Type of flower')
if train_or_test
    title(['Actual vs Desired Training Data Classification (Testing Fold: ',num2str(i),')'])
else
    title(['Actual vs Desired Testing Data Classification (Testing Fold: ',num2str(i),')'])
end
legend('Desired classification','Actual classification')
grid on

end


function [cM, accuracy] = confusionMatrix(desiredOutput,thresholdedActualOutput)
%  outputs the confusion matrix (as a table) and the accuracy in %
% A vectorized way to arrive at the confusion matrix 
a = desiredOutput; b = thresholdedActualOutput;
% diagel = diag(sum(a == b)); % count all elements that match
anm = a .* (a~= b); anm = anm(logical(sum(anm')),:); % make reduced matrices for elements that don't match
bnm = b .* (a~= b); bnm = bnm(logical(sum(bnm')),:);
otherel = anm' * bnm;  % a quick way to find which element was wrongly classified as which
diagel = diag(sum(a) - sum(otherel,2)'); % count all elements that match (by subtracting unmatched ones)
cM = diagel + otherel;

names = {'Set', 'Ver','Vir'};
cM = array2table(cM,'VariableNames', names, 'RowNames', names);

accuracy = sum(sum(diagel))/sum(sum(a)) * 100;
% disp(['Classification accuracy = ',num2str( accuracy)]);
end


function folds = crossValidation(numFolds, trainInput, trainOutput, testInput, testOutput)

randomTrainIndices = randperm(size(trainInput,1));
randomTestIndices = randperm(size(testInput,1));

randomizedTrainInput = trainInput;
randomizedTrainOutput = trainOutput;

randomizedTestInput = testInput;
randomizedTestOutput = testOutput;
    
folds = cell(numFolds,4);

aggregateInput = [randomizedTrainInput; randomizedTestInput];
aggregateOutput = [randomizedTrainOutput; randomizedTestOutput];

% print warning if this will not fold evenly
if mod(size(aggregateInput,1),numFolds) ~= 0
    error('This will not fold evenly');
end

foldSize = ceil(size(aggregateInput,1) / numFolds);

for i = 1:numFolds
    
    if i == numFolds
        maximumTestIndex = size(aggregateInput,1);
    else
        maximumTestIndex = i * foldSize;
    end
    
    minimumTestIndex = (i-1) * foldSize + 1;
        
    currentTrainInput = [aggregateInput(1:minimumTestIndex-1,:);
                         aggregateInput(maximumTestIndex+1:size(aggregateInput,1),:)];
    currentTrainOutput = [aggregateOutput(1:minimumTestIndex-1,:);
                          aggregateOutput(maximumTestIndex+1:size(aggregateInput,1),:)];

                      currentTestInput = aggregateInput(minimumTestIndex:maximumTestIndex,:);
    currentTestOutput = aggregateOutput(minimumTestIndex:maximumTestIndex,:);
    
    folds{i,1} = currentTrainInput;
    folds{i,2} = currentTrainOutput;
    folds{i,3} = currentTestInput;
    folds{i,4} = currentTestOutput;
    
end

end


