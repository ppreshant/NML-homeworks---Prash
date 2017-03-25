% “Exam01, Problem 1 (Compression)”
function BPvectorized
% Prashant Kalvapalle, Ragib Mostofa
% COMP 502, Spring 2017,Exam 01 Problem I
% Vectorized BP

batchSize = 1;  % set the size of the batch, i.e. number of patterns per batch
eval_points = 100; % number of points in the learning history or error vs time graph

numNodes = [64, 16, 64];  % set the number of nodes in each layers in the neural network including input layer - don't include bias nodes
weightMatrices = createWeightMatrices(numNodes,[0,.2]);  % create the weight matrices for each hidden layer and output layer; [] = range of the weights

learningRate = 3e-3;
alpha = 0.7; % forgetting rate for momentum term >> Make it 0 for no momentum correction 

tanhSlope = 1;  % set the slope of the hyperbolic tangent function

maxLearnSteps = 1e4;  % number of times each batch is processed ; can terminate before if converged
errorTolerance = 0.001;  % scaled error tolerance (for inputs between [.1 - 1])

%% Input and Output samples for TRAINING the BP network
load ocelot;

maxTrainScale = max(max(ocelot));  
% ocelotSc = ocelot ./ max(max(ocelot));  % scaling the input image pixels to lie between [0,1]

trainInput = loadVectors(ocelot); % each row is a pattern; size 768 x 64
trainOutput = trainInput;

scaledTrainInput = trainInput/maxTrainScale; % each row is a pattern; size 768 x 64
scaledTrainOutput = scaledTrainInput; % output is same as input

%% Input and Output samples for TESTING the BP network
load fruitstill;

maxTestScale =  max(max(fruitstill));
% fruitstillSc = fruitstill ./ max(max(fruitstill));   % scaling the input image pixels to lie between [0,1]

testInput = loadVectors(fruitstill);
testOutput = testInput;

scaledTestInput = testInput/maxTestScale;
scaledTestOutput = scaledTestInput; % output is same as input

%% calling the BPtraining algorithm
outDim = size(trainOutput);

[weightMatrices,otherVariables] = BPLearn(scaledTrainInput, scaledTrainOutput, scaledTestInput, scaledTestOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxLearnSteps, errorTolerance, alpha, eval_points,maxTrainScale,maxTestScale);

actualTrainOutput = BPrecall(scaledTrainInput, tanhSlope, numNodes, weightMatrices, outDim) .* maxTrainScale; % re-scaled
actualTestOutput = BPrecall(scaledTestInput, tanhSlope, numNodes, weightMatrices, outDim) .* maxTestScale;

% actualTrainOutput = sort(actualTrainOutput,'descend');
% disp(sort(actualTrainOutput,'descend'))

total_steps = otherVariables{1};
Erms_train = otherVariables{2}; %Erms_train(1,:) = maxTrainScale.*Erms_train(1,:); % scaling already in the error function
Erms_test = otherVariables{3}; %Erms_test(1,:) = maxTestScale.*Erms_test(1,:);

if total_steps == maxLearnSteps * batchSize
    disp(['Max iterations reached: MaxIters = ',num2str(total_steps)])
else
    disp(['LEARNING DONE: Steps taken = ',num2str(total_steps)])
end

disp(['Avg MSSE error = ',num2str(computeErrorMeasure(trainOutput,actualTrainOutput,1))]) % its already in data scale

%% reconstructing the image from the actual output from recall step

recalcTrain = reconstructImage(actualTrainOutput); % this is in original data scale 
recalcTest = reconstructImage(actualTestOutput);

ocelotBackCalc = reconstructImage(trainInput); % this is in original data scale
fruitstillBackCalc = reconstructImage(testInput);

%% plot for Training accuracy 
figure;

subplot(2,2,1); imagesc(ocelotBackCalc); colormap('gray'); title('Original Image'); colorbar
s2 = subplot(2,2,2); imagesc(recalcTrain); colormap('gray'); title('Reconstructed Image'); colorbar
s3 = subplot(2,2,3); imagesc(ocelotBackCalc - recalcTrain); colormap('gray'); title('Residual (difference) Image'); colorbar
% s3p = get(s3,'position');s2p = get(s2,'position'); s3p(3:4) = s2p(3:4); set(s3,'position',s3p); % setting image 3 same size as other images

%% plot for Testing accuracy 
% figure;
% 
% subplot(2,2,1); imagesc(fruitstillBackCalc); colormap('gray'); title('Original Image'); colorbar
% s2 = subplot(2,2,2); imagesc(recalcTest); colormap('gray'); title('Reconstructed Image'); colorbar
% s3 = subplot(2,2,3); imagesc(fruitstillBackCalc - recalcTest); colormap('gray'); title('Residual (difference) Image'); colorbar
% % s3p = get(s3,'position');s2p = get(s2,'position'); s3p(3:4) = s2p(3:4); set(s3,'position',s3p); % setting image 3 same size as other images

%% % plot for Training/Testing accuracy
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

%% plot Learning History
figure; plot(Erms_train(2,:),Erms_train(1,:)); %hold on;  plot(Erms_test(2,:),Erms_test(1,:));

grid on
xlabel('Learning Steps')
ylabel('Avg MSSE error : in original scale')
title('Learning History')
legend('Training Errors')%,'Testing Errors')

end


function [weightMatrices, otherVariables] = BPLearn(trainInput, trainOutput, testInput, testOutput, numNodes, weightMatrices, learningRate, tanhSlope, batchSize, maxLearnSteps, errorTolerance, alpha, eval_points,maxTrainScale,maxTestScale)
% The actual Neural Network in this function

%  For 3 layers (layer 1,2,3 - layer 1 is input)
% inputs go 1,2,3
% weights go 1,2  
% delta go 1 to 2

% NOTE: trainInput is a cell

%%
% Error initialization and other variables
total_steps = maxLearnSteps * batchSize;
otherVariables = cell(3,1); % for storing total_steps, Erms_store_train, Erms_store_test
eval_interval = maxLearnSteps / eval_points;
Erms_train = zeros(2,eval_points); Erms_test = Erms_train; % stores the RMS error every m iterations (m = eval_interval)
outDim = size(trainOutput);
% initial error
dum = 1; % dummy index for storing RMS errors at frequent intervals while training
frozenTrainOutput = BPrecall(trainInput, tanhSlope, numNodes, weightMatrices, outDim);
frozenTestOutput = BPrecall(testInput, tanhSlope, numNodes, weightMatrices, outDim);
Erms_train(1,dum) = computeErrorMeasure(trainOutput,frozenTrainOutput,maxTrainScale); Erms_train(2,dum) = 0;  % store errors and learning steps
Erms_test(1,dum) = computeErrorMeasure(testOutput,frozenTestOutput,maxTestScale); Erms_test(2,dum) = 0; dum = dum + 1; % store errors and learning steps

if batchSize > length(trainInput)
    disp('Batch size must be lower than or equal to the total number of available patterns. Please reset and retry!')
    return
end

oldWeightDeltas = createWeightDeltas(numNodes);
%% big loop
for i = 1:maxLearnSteps % big loop
    
%     numIters = 1e4;
%     learningRate = learningRate * ((i <= numIters/10) + .5 * (i > numIters/10 & i <= numIters/2.5) + .125 * (i > numIters/2.5 & i <= numIters*.8)+ .025 * (i > numIters*.8));
    randomIndices = randperm(size(trainInput,1));
    randomizedInput = trainInput(randomIndices,:);
    randomizedOutput = trainOutput(randomIndices,:);

    batchInput = randomizedInput(1:batchSize,:);
    batchOutput = randomizedOutput(1:batchSize,:);
    weightDeltas = createWeightDeltas(numNodes); %initializing deltaweights with 0s
    
%     learningRate = learningRate * (i <= 5e4) + (learningRate/2) * (i > 5e4) * (i <= 2e5) + (learningRate/20) * (i > 2e5);
    
    for k = 1:batchSize % loop over all patterns in the batch
        
%         nodeErrorGradients = createNodeValues(numNodes);
        layerOutputs = cell(1,length(numNodes));
        nodeDeltas = createNodeValues(numNodes);
        
        pattern = batchInput(k,:)'; % (64 x 1) =  column vector
        desiredOutput = batchOutput(k,:)';  % this is randomized don't use for testing % desired Output is a column vector (number x 1)
        
        % forward propagation
        layerOutputs{1} = pattern;
        layerOutputs{1}(end+1,1) = 1;  % fixing bias = 1
        
        for l = 1:length(numNodes)-2
            layerOutputs{l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{l});
            layerOutputs{l+1}(end) = 1; % fixing bias nodes = 1 before calculating next layer's output
        end
        l = l + 1; layerOutputs{l+1} = hyperbolicTangentFunction(tanhSlope, weightMatrices{l} * layerOutputs{l}); % for last layer since there is no bias
        
        % backward propagation
        m = length(numNodes);
        currentLayerOutput = layerOutputs{m};
        
        nodeDeltas{m-1} = diag(hyperbolicTangentDerivative(tanhSlope, currentLayerOutput)) * (desiredOutput - currentLayerOutput);
        for m = length(numNodes)-1:-1:2 % going over layers
                        
            previousLayerOutput = layerOutputs{m-1};

            % vectorizing the delta updates
            nodeDeltas{m-1} = computeTheNodeDeltas(nodeDeltas, tanhSlope, m-1, layerOutputs, weightMatrices);
            weightDeltas{m-1} = weightDeltas{m-1} + (learningRate * nodeDeltas{m-1} * previousLayerOutput');
        end
    end                             % end of the batch
    %         disp(weightMatrices{1})
    %         disp(weightMatrices{2})
    weightDeltas = updateWeights(weightDeltas, oldWeightDeltas, alpha);
    weightMatrices = updateWeights(weightMatrices, weightDeltas, 1);
    oldWeightDeltas = weightDeltas;
    
    if mod(i,eval_interval) == 0
        dum = i/eval_interval + 1;
        frozenTrainOutput = BPrecall(trainInput, tanhSlope, numNodes, weightMatrices, outDim);
        frozenTestOutput = BPrecall(testInput, tanhSlope, numNodes, weightMatrices, outDim);
        RMSE_train = computeErrorMeasure(trainOutput,frozenTrainOutput,maxTrainScale); % mean sq error => rescaling is squared
        RMSE_test = computeErrorMeasure(testOutput,frozenTestOutput,maxTestScale);
        Erms_train(1,dum) = RMSE_train; Erms_train(2,dum) = i*k;  % store errors and learning steps
        Erms_test(1,dum) = RMSE_test; Erms_test(2,dum) = i*k; % store errors and learning steps
        
        if RMSE_train < errorTolerance
            total_steps = (i)* batchSize; % steps taken to complete the training
            Erms_train = Erms_train(:,Erms_train(1,:) ~= 0); Erms_test = Erms_test(:,Erms_test(1,:) ~= 0); % clip the Error storage matrix when terminating
            otherVariables{1} = total_steps; otherVariables{2} = Erms_train; otherVariables{3} = Erms_test; % giving output variables
            return
        end
    end
            
end

otherVariables{1} = total_steps;
otherVariables{2} = Erms_train;
otherVariables{3} = Erms_test;

end


function hiddenNodeDeltas = computeTheNodeDeltas(nodeDeltas, tanhSlope, layerIndex,layerOutputs, weightMatrices)
% enters into this function to calculate nodeDeltas recursively for each
% layer in the loop
nextLayerWeightVectorTranspose = weightMatrices{layerIndex+1}';
nextLayerDeltaVector = nodeDeltas{layerIndex+1};

derivative = hyperbolicTangentDerivative(tanhSlope, layerOutputs{layerIndex+1});

hiddenNodeDeltas = diag(derivative) * nextLayerWeightVectorTranspose * nextLayerDeltaVector;

end


function testOutput = BPrecall(testInput, tanhSlope, numNodes, weightMatrices, outDim)
% recall function
testOutput = zeros(outDim); % creating a dummy output matrix -to desired output dimensions

for i = 1:length(testInput)
    output = [testInput(i,:),1]'; % a temp output variable for each input vector
    for j = 1:length(numNodes) - 1
        output = hyperbolicTangentFunction(tanhSlope,weightMatrices{j} * output);
        if j ~= length(numNodes) - 1 % can remove 'if' and make the loop run till length - 2 and have the last output after the for
            output(end) = 1;
        end
    end
    testOutput(i,:) = output; % stores the output vector for every input vector
end

end


function updatedWeights = updateWeights(weightMatrices, weightDeltas, alpha)
% updates weights using deltaW input
updatedWeights = weightMatrices; % dummy initialization with same dimension as existing weights

for i = 1:length(weightMatrices)
    updatedWeights{i} = weightMatrices{i} + alpha .* weightDeltas{i};
end

end


function MSSE = computeErrorMeasure(desiredOutput, actualOutput, scale)
% finding root mean square error per pattern
% Input are matrices 768 x 64

MSSE = norm(((desiredOutput - actualOutput) * scale),'fro').^2 / numel(desiredOutput);
% RMSE = sqrt(sum((desiredOutput - actualOutput) .^ 2) ./ size(desiredOutput,1));

end


function weightMatrices = createWeightMatrices(numNodes, weightRange)
% create random entries in the weight matrix: weight scale decides range of
% weights
numMatrices = length(numNodes) - 1;
weightMatrices = cell(1, numMatrices);

for j = 1:numMatrices
    weightMatrices{j} = rand(numNodes(j+1), numNodes(j)+1) .* (weightRange(2) - weightRange(1)) + weightRange(1) ;
    if j ~= numMatrices
        weightMatrices{j}(end+1,:) = zeros(1,length(weightMatrices{j}(end,:)));
    end
end

end


function weightDeltas = createWeightDeltas(numNodes)
% creating dummy zero vectors for weight deltas
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
% creating dummy zero values for nodeDeltas
numLayers = length(numNodes);
nodeValues = cell(1, numLayers - 1);

for j = 2:numLayers
    nodeValues{j-1} = zeros(1,numNodes(j));
end

end


function f = hyperbolicTangentDerivative(a,fx)

f = a .* (1 - fx .^ 2);

end


function f = hyperbolicTangentFunction(a,x)

xp = 2*a.*x;

f = (exp(xp) -1) ./ (exp(xp) + 1);

end




