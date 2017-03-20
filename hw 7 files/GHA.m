% hw7 Problem 2.2
function [trainEigenVec] = GHA
% Prashant Kalvapalle 
% Comp 504 HW6 - Base code for all problems

numIters = 50000;
alpha = 1e-3;
numOutputPE = 4; %  = number of eigen vectors needed?

% Input data entry
trainDataFilename = 'iris-train copy.txt';
testDataFilename = 'iris-test copy.txt';

[trainInput,trainOutput] = loadData(trainDataFilename); % trainInput = 75 x 4
% [testInput,testOutput] = loadData(testDataFilename);
% dataInput = [trainInput ; testInput]';
% dataClasses = [trainOutput ; testOutput];

% zero mean preprocessing
trainInput = trainInput - mean(trainInput);

% finding eigen vectors without ANN
eigenCheck = pca(trainInput);

initWeightMatrix = rand(numOutputPE,size(trainInput,2));  % create the weight matrices for each hidden layer and output layer

% calling sanger's generalized heabbian algo
[trainEigenVec,storedValues] = sangGHA(trainInput,alpha,numIters,initWeightMatrix);
trainEigenVec * trainEigenVec'
figure; plot(0:500:numIters,storedValues)
end

function [trainEigenVec,storedValues] = sangGHA(trainInput,alpha,numIters,initWeightMatrix)
% implementing sanger's generalized heabbian algo
% trainInput = 75 x 4
weightMatrix = initWeightMatrix; % initialization - dimension 1 x 4

% to store the P * P' which is expected to be I
storedValues = zeros(numIters/500,size(weightMatrix,2)); %storedValues(1,:) = weightMatrix(1,:); dum = 2;
checkI = weightMatrix * weightMatrix' ;%- eye(size(weightMatrix,1));
storedValues(1,:) = diag(checkI); dum = 2;
        
for i = 1:numIters
    x = trainInput(randi(size(trainInput,1)),:)'; % pick a random pattern - vector of 4 x 1
    y = weightMatrix * x;
    crossCorrel = y * y';
    lowerTriangular = makeLowerTriangular(crossCorrel);
    
    % weight update step
    weightMatrix = weightMatrix + alpha*(y*x' - lowerTriangular * weightMatrix);
    
    % to make learning history
   
    if ~mod(i,500)
        checkI = weightMatrix * weightMatrix' ;%- eye(size(weightMatrix,1)); 
        storedValues(dum,:) = diag(checkI); dum = dum + 1;
%         storedValues(dum,:) = weightMatrix(1,:); dum = dum + 1;  % this thing is only for 1 output PE
    end

end

trainEigenVec = weightMatrix;
end


function lowerTriangular = makeLowerTriangular(crossCorrel)
% to make the correlation matrix lower triangular
[m,n] = size(crossCorrel);
rowM = repmat((1:m)',1,n); colM = repmat(1:n,m,1);
lowerTriangular = crossCorrel;
lowerTriangular(colM > rowM) = 0;
   
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

