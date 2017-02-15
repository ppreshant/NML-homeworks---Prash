function ProblemII
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I, ProblemI
% 

batchSize = 10;  % set the size of the batch, i.e. number of patterns per batch

numNodes = [1, 10, 1];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightMatrices(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 0.05;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxIterations = 3000;
errorTolerance = 0.08;

N_training_pts = 50; % number of training patterns

trainInput = linspace(0.1,1.0,N_training_pts)';
trainOutput = multiplicativeInverseFunction(trainInput);

maxTrainScale = max(trainOutput);
% scaledTrainInput = trainInput .* maxTrainScale;
scaledTrainOutput = trainOutput ./ maxTrainScale;

testInput = linspace(0.1,1.0,N_training_pts/2)';
testOutput = multiplicativeInverseFunction(testInput);

maxTestScale = max(testOutput);
% scaledTestInput = testInput .* maxTestScale; % dont have to scale input
scaledTestOutput = testOutput ./ maxTestScale;

weightMatrices = train(trainInput, scaledTrainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);

% actualTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale;
actualTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale; % re-scaled
disp(actualTrainOutput)
disp(['RMS error = ',num2str(computeRMSE(trainOutput,actualTrainOutput))])

% plot for training vs testing
% figure
% hold on
% grid on
% 
% plot(testInput,actualTestOutput);
% hold on;
% plot(testInput,testOutput);
% 
% xlabel('x')
% ylabel('f(x) = 1/x')
% title('Comparison of training and testing accuracies')

% plot for testing accuracy
figure
hold on
grid on

subplot(2,1,1)
plot(trainInput,actualTrainOutput);
hold on;
plot(trainInput,trainOutput);

xlabel('x')
ylabel('f(x) = 1/x')
title('Comparison of training accuracy wrt desired output')

subplot(2,1,2)
plot(trainInput,abs(trainOutput - actualTrainOutput));

xlabel('x')
ylabel('D - actual Y')
title('training output - desired output')
end


function weightMatrices = train(trainInput, trainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)

% The actual neural network in this function

if batchSize > length(trainInput)
    disp('Batch size must be lower than or equal to the total number of available patterns. Please reset and retry!')
    return
end

numBatches = ceil(length(trainInput) ./ batchSize);

for i = 1:maxIterations % big loop
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);

    batchInput = randomizedInput(1:batchSize,:);
    batchOutput = randomizedOutput(1:batchSize,:);
    weightDeltas = createWeightDeltas(numNodes); %initializing deltaweights with 0s
    
    for k = 1:batchSize % loop over all patterns in the batch
        
        nodeErrorGradients = createNodeValues(numNodes);
        layerOutputs = cell(1,length(numNodes));
        nodeDeltas = createNodeValues(numNodes);
        
        pattern = batchInput(k,:);
        desiredOutput = batchOutput(k,:);  % this is randomized don't use for testing
        
        % forward propagation
        layerOutputs{1} = pattern;
        layerOutputs{1}(end+1) = 1;  % fixing bias = 1
        
        for l = 1:length(numNodes)-2
            layerOutputs{l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{l}')';
            layerOutputs{l+1}(end) = 1; % fixing bias nodes = 1 before calculating next layer's output
        end
        l = l + 1; layerOutputs{l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{l}')'; % for last layer since there is no bias
        
        % backward propagation
        for m = length(numNodes):-1:2 % going over layers
            
            currentLayerOutput = layerOutputs{m};
            previousLayerOutput = layerOutputs{m-1};
            
            % can we change into matrix multiplication - for n and for p  - too many loops over here
            for n = 1:length(currentLayerOutput) % for each layer going over nodes
                
                for p = 1:length(previousLayerOutput) % need summation over nodes of previous layer
                    
                    if m == length(numNodes) % delta rule for last layer  % remove this if and generalize if possible
                        nodeDeltas{m-1}(n) = (desiredOutput(n) - currentLayerOutput(n)) .* hyperbolicTangentDerivative(tanhSlope, weightMatrices{m-1}(n,:) * previousLayerOutput')';
                        nodeErrorGradients{m-1}(n) = -1 .* nodeDeltas{m-1}(n) .* previousLayerOutput(p);
                        weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (-learningRate .* nodeErrorGradients{m-1}(n));
                    else
                        if n ~= length(currentLayerOutput)
                            nodeErrorGradients{m-1}(n) = computeHiddenNodeErrorGradient(m-1, n, nodeDeltas, weightMatrices);
                            nodeDeltas{m-1}(n) = computeHiddenNodeDelta(nodeErrorGradients{m-1}(n), tanhSlope, m-1, n, layerOutputs, weightMatrices);
                            weightDeltas{m-1}(n,p) = weightDeltas{m-1}(n,p) + (learningRate .* nodeDeltas{m-1}(n) .* previousLayerOutput(p)); % check if the formula is right
                        end
                    end
                end
            end
        end
    end
    %         disp(weightMatrices{1})
    %         disp(weightMatrices{2})
    weightMatrices = updateWeights(numNodes, weightMatrices, weightDeltas);
        
end

end


function testOutput = test(testInput, tanhSlope, numNodes, weightMatrices)

testOutput = zeros(length(testInput),1);

for i = 1:length(testInput)
    output = [testInput(i,:),1]';
    for j = 1:length(numNodes) - 1
        output = hyperbolicTangentFunction(tanhSlope,weightMatrices{j} * output);
        if j ~= length(numNodes) - 1 % can remove 'if' and make the loop run till length - 2 and have the last output after the for
            output(end) = 1;
        end
    end
    testOutput(i) = output;
end

end


function updatedWeights = updateWeights(numNodes, weightMatrices, weightDeltas)

updatedWeights = weightMatrices; % dummy initialization with same dimension as existing weights

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


