function ProblemII
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I, ProblemI
% 
batchSize = 200;

numNodes = [1, 10, 1];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightMatrices(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 0.05;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxIterations = 3000;
errorTolerance = 0.08;

trainInput = linspace(0.1,1.0,200)';

trainOutput = multiplicativeInverseFunction(linspace(0.1,1.0,200)');
maxTrainScale = max(trainOutput);
scaledTrainOutput = trainOutput ./ maxTrainScale;

testInput = linspace(0.1,1.0,100)';

testOutput = multiplicativeInverseFunction(linspace(0.1,1.0,100)');
maxTestScale = max(testOutput);
scaledTestOutput = testOutput ./ maxTestScale;

[weightMatrices,total_steps, Erms_store_train, Erms_store_test] = train(trainInput, scaledTrainOutput, testInput, scaledTestOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);
% Recall step
actualTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices) .* maxTrainScale;
actualTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale;

RMSe = norm(testOutput - actualTestOutput)/sqrt(size(testOutput,1)); % calculates the RMS error for all patterns

if total_steps == maxIterations * size(actualTestOutput,1)
      disp(['Max iterations reached; Max iters =',num2str(total_steps)])
else
    disp(['LEARNING DONE: Steps taken = ',num2str(total_steps)])
end
% disp(Erms_store_test)
disp(['RMS error = ',num2str(RMSe)])

figure(1); 
hold on
grid on

plot(Erms_store_train(2,:),Erms_store_train(1,:));  % plot train and testing errors
%plot(Erms_store_test(2,:),Erms_store_test(1,:));

title('RMS Training Error VS Training Steps');
xlabel('Number of Training Steps'); 
ylabel('RMS of Errors of all Patterns')

figure(2)
hold on
grid on

plot(trainInput,trainOutput)
plot(trainInput,actualTrainOutput)

plot(testInput,testOutput)
plot(testInput,actualTestOutput)

xlabel('x')
ylabel('f(x) = 1/x')
title('Comparison of training and testing accuracies')
legend('Desired Training Curve','Actual Training Curve','Desired Test Curve','Actual Testing Curve')

end


function [weightMatrices, total_steps, Erms_store_train, Erms_store_test] = train(trainInput, trainOutput, testInput, testOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)

% The actual neural network in this function

total_steps = maxIterations * size(trainInput,1); % default total steps until convergence - changed later after test condition

eval_interval = total_steps/50;
Erms_store_train = zeros(2,20); % stores the RMS error every m iterations
Erms_store_test = zeros(2,20);
% initial error
frozenTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);  % compute test and train errors
frozenTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices);
RMSe_train = norm(trainOutput - frozenTrainOutput)/sqrt(size(trainOutput,1));
RMSe_test = norm(testOutput - frozenTestOutput)/sqrt(size(testOutput,1));
dum = 1; Erms_store_train(1,dum) = RMSe_train; Erms_store_train(2,dum) = 0; dum = dum + 1; % store errors and learning steps
Erms_store_test(1,dum) = RMSe_test; Erms_store_test(2,dum) = 0;

if batchSize > length(trainInput)
    disp('Batch size must be lower than or equal to the total number of available patterns. Please reset and retry!')
    return
end

numBatches = ceil(length(trainInput) ./ batchSize);

for i = 1:maxIterations
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);
    
    for j = 1:numBatches
        
        weightDeltas = createWeightDeltas(numNodes);
        layerOutputs = cell(batchSize,length(numNodes));
        nodeDeltas = createNodeValues(batchSize,numNodes);
        
        if j * batchSize > length(randomizedInput)
            batchInput = randomizedInput((j-1) * batchSize + 1:end,:);
            batchOutput = randomizedOutput((j-1) * batchSize + 1:end,:);
        else
            batchInput = randomizedInput((j-1) * batchSize + 1:j * batchSize,:);
            batchOutput = randomizedOutput((j-1) * batchSize + 1:j * batchSize,:);
        end
        
        for k = 1:size(batchInput,1)
            
            nodeErrorGradients = createNodeValues(batchSize,numNodes);
            
            pattern = batchInput(k,:);
            desiredOutput = batchOutput(k,:);  % this is randomized don't use for testing
            
            % forward propagation
            layerOutputs{k,1} = pattern;
            layerOutputs{k,1}(end+1) = 1;
            
            for l = 1:length(numNodes)-1
                layerOutputs{k,l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{k,l}')';
                if l ~= length(numNodes) - 1
                    layerOutputs{k,l+1}(end) = 1;
                end
            end
            
            % backward propagation
            for m = length(numNodes):-1:2
                
                currentLayerOutput = layerOutputs{k,m};
                previousLayerOutput = layerOutputs{k,m-1};
                
                for n = 1:length(currentLayerOutput)
                    
                    for p = 1:length(previousLayerOutput)
                        
                        if m == length(numNodes)
                            nodeDeltas{k,m-1}(n) = (desiredOutput(n) - currentLayerOutput(n)) .* hyperbolicTangentDerivative(tanhSlope, weightMatrices{m-1}(n,:) * previousLayerOutput')';
                            nodeErrorGradients{k,m-1}(n) = -1 .* nodeDeltas{k,m-1}(n) .* previousLayerOutput(p);
                            weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (-learningRate .* nodeErrorGradients{k,m-1}(n));
                        else
                            if n ~= length(currentLayerOutput)
                                nodeErrorGradients{k,m-1}(n) = computeHiddenNodeErrorGradient(k, m-1, n, nodeDeltas, weightMatrices);
                                nodeDeltas{k,m-1}(n) = computeHiddenNodeDelta(nodeErrorGradients{m-1}(n), tanhSlope, k, m-1, n, layerOutputs, weightMatrices);
                                weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (learningRate .* nodeDeltas{k,m-1}(n) .* previousLayerOutput(p));
                            end
                        end
                    end
                end
            end
        end
        weightMatrices = updateWeights(numNodes, weightMatrices, weightDeltas);
        % disp(weightMatrices{1})
        % disp(weightMatrices{1})
        
    end
    frozenTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);
    frozenTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices);
    RMSE_train = computeRMSE(trainOutput,frozenTrainOutput);
    RMSE_test = computeRMSE(testOutput,frozenTestOutput);
    
    if RMSE_train < errorTolerance
        total_steps = (i-1)* size(trainInput,1) + j; % steps taken to complete the training
        Erms_store_train = Erms_store_train(:,Erms_store_train(1,:) ~= 0); % clip the Error storage matrix when terminating
        return
    end
    
    if mod(i*j,eval_interval) == 0
        Erms_store_train(1,dum) = RMSE_train; Erms_store_train(2,dum) = i*j; dum = dum + 1; % store errors and learning steps
        Erms_store_test(1,dum) = RMSE_test; Erms_store_test(2,dum) = i*j; % store errors and learning steps
    end
end

end


function testOutput = test(testInput, tanhSlope, numNodes, weightMatrices)

testOutput = zeros(length(testInput),1);

for i = 1:length(testInput)
    output = [testInput(i,:),1]';
    for j = 1:length(numNodes) - 1
        output = hyperbolicTangentFunction(tanhSlope,weightMatrices{j} * output);
        if j ~= length(numNodes) - 1
            output(end) = 1;
        end
    end
    testOutput(i) = output;
end

end


function updatedWeights = updateWeights(numNodes, weightMatrices, weightDeltas)

updatedWeights = createWeightMatrices(numNodes);

for i = 1:length(weightMatrices)
    updatedWeights{i} = weightMatrices{i} + weightDeltas{i};
end

end


function hiddenNodeDelta = computeHiddenNodeDelta(hiddenNodeErrorGradient, tanhSlope, patternIndex, layerIndex, nodeIndex, layerOutputs, weightMatrices)

previousLayerOutput = layerOutputs{patternIndex,layerIndex};
currentLayerWeightVector = weightMatrices{layerIndex}(nodeIndex,:);
derivative = hyperbolicTangentDerivative(tanhSlope, currentLayerWeightVector * previousLayerOutput');

hiddenNodeDelta = -1 .* derivative .* hiddenNodeErrorGradient;

end


function hiddenNodeErrorGradient = computeHiddenNodeErrorGradient(patternIndex, layerIndex, nodeIndex, nodeDeltas, weightMatrices)
% unnecessary function??
nextLayerWeightVector = weightMatrices{layerIndex+1}(:,nodeIndex);
deltaVector = nodeDeltas{patternIndex,layerIndex+1};
summation = deltaVector * nextLayerWeightVector;

hiddenNodeErrorGradient = -summation; % this formula looks wrong to me - prashant

end


function RMSE = computeRMSE(desiredOutput, actualOutput)

RMSE = sqrt(sum((desiredOutput - actualOutput) .^ 2) / length(desiredOutput));

end


function weightMatrices = createWeightMatrices(numNodes)

numMatrices = length(numNodes) - 1;
weightMatrices = cell(1, numMatrices);

for j = 1:numMatrices
    weightMatrices{j} = rand(numNodes(j+1), numNodes(j)+1) * 0.2 - 0.1;
    if j ~= numMatrices
        weightMatrices{j}(end+1,:) = zeros(1,length(weightMatrices{j}(end,:)));
    end
end

end


function weightDeltas = createWeightDeltas(numNodes)

numMatrices = length(numNodes) - 1;
weightDeltas = cell(1, numMatrices);

for i = 1:numMatrices
    weightDeltas{i} = zeros(numNodes(i+1), numNodes(i)+1);
    if i ~= numMatrices
        weightDeltas{i}(end+1,:) = zeros(1,length(weightDeltas{i}(end,:)));
    end
end

end


function nodeValues = createNodeValues(batchSize,numNodes)

numLayers = length(numNodes);
nodeValues = cell(batchSize, numLayers - 1);

for i = 1:batchSize
    for j = 2:numLayers
        nodeValues{i,j-1} = zeros(1,numNodes(j));
    end
end

end


function f = hyperbolicTangentDerivative(a,x)

f = a .* (1 - hyperbolicTangentFunction(a,x) .^ 2);

end


function f = hyperbolicTangentFunction(a,x)

f = (exp(a .* x) - exp(-a .* x)) ./ (exp(a .* x) + exp(-a .* x));

end


function f = multiplicativeInverseFunction(x)

f = 1 ./ x;

end


