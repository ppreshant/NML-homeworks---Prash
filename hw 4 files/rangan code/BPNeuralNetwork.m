function BPNeuralNetwork
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I, ProblemI
% 
numNodes = [3, 3, 1];  % set the number of nodes and layers in the neural network
weightMatrices = createWeightValues(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 0.01;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

batchSize = 1;

maxIterations = 1000;
errorTolerance = 0.15;

input = [-1, -1, 1;
         -1,  1, 1;
          1, -1, 1;
          1,  1, 1];

output = [-1;
           1; 
           1; 
          -1];
% train the XOR
[weightMatrices,total_steps] = train(input, output, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);

testOutput = test(input, tanhSlope, numNodes, weightMatrices);
RMSe = norm(output - testOutput)/sqrt(size(output,1));
disp(testOutput)
disp(['LEARNING DONE: Steps taken = ',num2str(total_steps)])
disp(['RMS error = ',num2str(RMSe)])
end


function [weightMatrices,total_steps] = train(trainInput, trainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)
% The actual neural network in this function
total_steps = maxIterations * size(trainInput,1); % default total steps until convergence - changed later after test condition

layerOutputs = cell(1,length(numNodes));

nodeDeltas = createNodeValues(numNodes);
nodeErrorGradients = createNodeValues(numNodes);
weightDeltas = createWeightValues(numNodes);

for i = 1:maxIterations
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);
    for j = 1:size(trainInput,1)
        pattern = randomizedInput(j,:);
        desiredOutput = randomizedOutput(j,:);
        
        % forward propagation
        layerOutputs{1} = pattern;
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
        
        testOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);
        desiredOutput = trainOutput;
        RMSE = computeRMSE(desiredOutput,testOutput);
        if RMSE < errorTolerance
            total_steps = (i-1)* size(trainInput,1) + j; % steps taken to complete the training
            
            return
        end
    end
end

end


function testOutput = test(testInput, tanhSlope, numNodes, weightMatrices)

testOutput = zeros(length(testInput),1);

for i = 1:length(testInput)
    output = testInput(i,:)';
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
% unnecessary function??
nextLayerWeightVector = weightMatrices{layerIndex+1}(:,nodeIndex);
deltaVector = nodeDeltas{layerIndex+1};
summation = deltaVector * nextLayerWeightVector;

hiddenNodeErrorGradient = -summation; % this formula looks wrong to me - prashant

end


function RMSE = computeRMSE(desiredOutput, actualOutput)

RMSE = sqrt(sum((desiredOutput - actualOutput) .^ 2) / length(desiredOutput));

end


function weightValues = createWeightValues(numNodes)

numMatrices = length(numNodes) - 1;
weightValues = cell(1, numMatrices);

for i = 1:numMatrices
    weightValues{i} = rand(numNodes(i+1), numNodes(i)) * 2 - 1;
    %weightValues{i}(end,:) = zeros(1,length(weightValues{i}(end,:)));
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

function d = documnt(Initial_weights,tanh_slope,learning_parameter,stopping_critereon)
f = fopen(BP_output.txt,w);

t1 = table(tanh_slope,learning_parameter,stopping_critereon);
t2 = table()
fprintf(Initial_weights)
formatspec = '';
end

