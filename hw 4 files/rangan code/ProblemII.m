function BPNeuralNetwork
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I, ProblemI
% 

batchSize = 1;  % set the size of the batch, i.e. number of patterns per batch

numNodes = [1, 10, 1];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightMatrices(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 0.05;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxIterations = 10000;
errorTolerance = 0.08;

trainInput = linspace(0.1,1.0,200)';
trainOutput = multiplicativeInverseFunction(linspace(0.1,1.0,200)');

maxTrainScale = max(trainOutput);
% scaledTrainInput = trainInput .* maxTrainScale;
scaledTrainOutput = trainOutput ./ maxTrainScale;

testInput = linspace(0.1,1.0,100)';
testOutput = multiplicativeInverseFunction(linspace(0.1,1.0,100)');

maxTestScale = max(testOutput);
% scaledTestInput = testInput .* maxTestScale;
scaledTestOutput = testOutput ./ maxTestScale;

weightMatrices = train(trainInput, scaledTrainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);

actualTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale;

disp(actualTestOutput)

figure
hold on
grid on

plot(testInput,actualTestOutput)

xlabel('x')
ylabel('f(x) = 1/x')
title('Comparison of training and testing accuracies')

end


function weightMatrices = train(trainInput, trainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)

% The actual neural network in this function

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
                
        if j * batchSize > length(randomizedInput)
            batchInput = randomizedInput((j-1) * batchSize + 1:end,:);
            batchOutput = randomizedOutput((j-1) * batchSize + 1:end,:);
        else
            batchInput = randomizedInput((j-1) * batchSize + 1:j * batchSize,:);
            batchOutput = randomizedOutput((j-1) * batchSize + 1:j * batchSize,:);
        end
        
        weightDeltas = createWeightDeltas(numNodes);
        
        for k = 1:size(batchInput,1)
            
            nodeErrorGradients = createNodeValues(numNodes);
            layerOutputs = cell(1,length(numNodes));
            nodeDeltas = createNodeValues(numNodes);
            
            pattern = batchInput(k,:);
            desiredOutput = batchOutput(k,:);  % this is randomized don't use for testing
            
            % forward propagation
            layerOutputs{1} = pattern;
            layerOutputs{1}(end+1) = 1;
            
            for l = 1:length(numNodes)-1
                layerOutputs{l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{l}')';
                if l ~= length(numNodes) - 1
                    layerOutputs{l+1}(end) = 1;
                end
            end
            
            % backward propagation
            for m = length(numNodes):-1:2
                
                currentLayerOutput = layerOutputs{m};
                previousLayerOutput = layerOutputs{m-1};
                
                for n = 1:length(currentLayerOutput)
                    
                    for p = 1:length(previousLayerOutput)
                        
                        if m == length(numNodes)
                            nodeDeltas{m-1}(n) = (desiredOutput(n) - currentLayerOutput(n)) .* hyperbolicTangentDerivative(tanhSlope, weightMatrices{m-1}(n,:) * previousLayerOutput')';
                            nodeErrorGradients{m-1}(n) = -1 .* nodeDeltas{m-1}(n) .* previousLayerOutput(p);
                            weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (-learningRate .* nodeErrorGradients{m-1}(n));
                        else
                            if n ~= length(currentLayerOutput)
                                nodeErrorGradients{m-1}(n) = computeHiddenNodeErrorGradient(m-1, n, nodeDeltas, weightMatrices);
                                nodeDeltas{m-1}(n) = computeHiddenNodeDelta(nodeErrorGradients{m-1}(n), tanhSlope, m-1, n, layerOutputs, weightMatrices);
                                weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (learningRate .* nodeDeltas{m-1}(n) .* previousLayerOutput(p));
                            end
                        end
                    end
                end
            end
        end        
%         disp(weightMatrices{1})
%         disp(weightMatrices{2})
        
    end
    weightMatrices = updateWeights(numNodes, weightMatrices, weightDeltas);
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


function hiddenNodeDelta = computeHiddenNodeDelta(hiddenNodeErrorGradient, tanhSlope, layerIndex, nodeIndex, layerOutputs, weightMatrices)

previousLayerOutput = layerOutputs{layerIndex};
currentLayerWeightVector = weightMatrices{layerIndex}(nodeIndex,:);
derivative = hyperbolicTangentDerivative(tanhSlope, currentLayerWeightVector * previousLayerOutput');

hiddenNodeDelta = -1 .* derivative .* hiddenNodeErrorGradient;

end


function hiddenNodeErrorGradient = computeHiddenNodeErrorGradient(layerIndex, nodeIndex, nodeDeltas, weightMatrices)
% unnecessary function??
nextLayerWeightVector = weightMatrices{layerIndex+1}(:,nodeIndex);
deltaVector = nodeDeltas{layerIndex+1};
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
    weightMatrices{j} = rand(numNodes(j+1), numNodes(j)+1);
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


function nodeValues = createNodeValues(numNodes)

numLayers = length(numNodes);
nodeValues = cell(1, numLayers - 1);

for j = 2:numLayers
    nodeValues{j-1} = zeros(1,numNodes(j));
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


