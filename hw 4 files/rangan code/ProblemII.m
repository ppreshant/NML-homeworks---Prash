function ProblemII
% Ragib Mostofa, COMP 502, Spring 2017, Homework Assignment IV Part I, ProblemI
% 

batchSize = 200;  % set the size of the batch, i.e. number of patterns per batch

numNodes = [1, 10, 1];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightMatrices(numNodes);  % create the weight matrices for each hidden layer and output layer

learningRate = 1/batchSize;

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxIterations = 10000;
errorTolerance = 0.08;

N_training_pts = 200; % number of training patterns

trainInput = linspace(0.11,1.0,N_training_pts)';
trainOutput = multiplicativeInverseFunction(trainInput);

maxTrainScale = max(trainOutput);
% scaledTrainInput = trainInput .* maxTrainScale;
scaledTrainOutput = trainOutput ./ maxTrainScale;

testInput = rand(N_training_pts/2,1) * 0.9 + 0.1;
testOutput = multiplicativeInverseFunction(testInput);

maxTestScale = max(testOutput);
% scaledTestInput = testInput .* maxTestScale; % dont have to scale input
scaledTestOutput = testOutput ./ maxTestScale;

[weightMatrices,otherVariables] = train(trainInput, scaledTrainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance);

% actualTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale;
actualTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices) .* maxTestScale; % re-scaled
% actualTrainOutput = sort(actualTrainOutput,'descend');
disp(sort(actualTrainOutput,'descend'))
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

plot(sort(trainInput,'descend'),sort(actualTrainOutput),'--');
hold on;
plot(trainInput,trainOutput);

xlabel('x')
ylabel('f(x) = 1/x')
title('Comparison of training accuracy wrt desired output')
legend('Learnt Function','Actual Function')
% plot(trainInput,abs(trainOutput - actualTrainOutput));
% 
% xlabel('x')
% ylabel('RMS error : calculated from scaled error of all train/test patterns')
% title('Learning History')
end


function [weightMatrices, otherVariables] = train(trainInput, trainOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxIterations, errorTolerance)

% The actual neural network in this function

%  For 3 layers (layer 1,2,3 - layer 1 is input)
% inputs go 1,2,3
% weights go 1,2  
% delta go 1 to 2

otherVariables = cell(3,1); % for storing total_steps, Erms_store_train, Erms_store_test

if batchSize > length(trainInput)
    disp('Batch size must be lower than or equal to the total number of available patterns. Please reset and retry!')
    return
end

for i = 1:maxIterations % big loop
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);

    batchInput = randomizedInput(1:batchSize,:);
    batchOutput = randomizedOutput(1:batchSize,:);
    weightDeltas = createWeightDeltas(numNodes); %initializing deltaweights with 0s
    
    for k = 1:batchSize % loop over all patterns in the batch
        
%         nodeErrorGradients = createNodeValues(numNodes);
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
        m = length(numNodes);
        currentLayerOutput = layerOutputs{m};
        previousLayerOutput = layerOutputs{m-1};
        
        nodeDeltas{m-1} = diag(hyperbolicTangentDerivative(tanhSlope, weightMatrices{m-1} * previousLayerOutput')) * (desiredOutput - currentLayerOutput);
        for m = length(numNodes)-1:-1:2 % going over layers
                        
            previousLayerOutput = layerOutputs{m-1};

            % vectorizing the delta updates
            nodeDeltas{m-1} = computeTheNodeDeltas(nodeDeltas, tanhSlope, m-1, layerOutputs, weightMatrices);
            weightDeltas{m-1} = weightDeltas{m-1} + (learningRate * nodeDeltas{m-1} * previousLayerOutput);
        end
    end                             % end of the batch
    %         disp(weightMatrices{1})
    %         disp(weightMatrices{2})
    weightMatrices = updateWeights(weightMatrices, weightDeltas);
    
%     frozenTrainOutput = test(trainInput, tanhSlope, numNodes, weightMatrices);
%     frozenTestOutput = test(testInput, tanhSlope, numNodes, weightMatrices);
%     RMSE_train = computeRMSE(trainOutput,frozenTrainOutput);
%     RMSE_test = computeRMSE(testOutput,frozenTestOutput);
%     
%     if RMSE_train < errorTolerance
%         total_steps = (i-1)* size(trainInput,1) + j; % steps taken to complete the training
%         Erms_store_train = Erms_store_train(:,Erms_store_train(1,:) ~= 0); % clip the Error storage matrix when terminating
%         return
%     end
%     
%     if mod(i*j,eval_interval) == 0
%         Erms_store_train(1,dum) = RMSE_train; Erms_store_train(2,dum) = i*j; dum = dum + 1; % store errors and learning steps
%         Erms_store_test(1,dum) = RMSE_test; Erms_store_test(2,dum) = i*j; % store errors and learning steps
%     end
end

end


function hiddenNodeDeltas = computeTheNodeDeltas(nodeDeltas, tanhSlope, layerIndex,layerOutputs, weightMatrices)
% enters into this function to calculate nodeDeltas recursively for each
% layer in the loop
previousLayerOutput = layerOutputs{layerIndex};
currentLayerWeightVector = weightMatrices{layerIndex};
nextLayerWeightVectorTranspose = weightMatrices{layerIndex+1}';
nextLayerDeltaVector = nodeDeltas{layerIndex+1};

derivative = hyperbolicTangentDerivative(tanhSlope, currentLayerWeightVector * previousLayerOutput');

hiddenNodeDeltas = diag(derivative) * nextLayerWeightVectorTranspose * nextLayerDeltaVector;

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


function updatedWeights = updateWeights(weightMatrices, weightDeltas)

updatedWeights = weightMatrices; % dummy initialization with same dimension as existing weights

for i = 1:length(weightMatrices)
    updatedWeights{i} = weightMatrices{i} + weightDeltas{i};
end

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


