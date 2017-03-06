% hw6 Problem 1
function SOM1
% Prashant Kalvapalle 
% Comp 504 HW6 - Base code for all problems

% NOTE : Initial and final lattice is a cell representation, In the function it is
% used as a multi-dimensional matrix

latticeSize = [10 10]; 
initRadius = max(latticeSize); % Initial radius of influence

numIters = 50000; % number of learning steps
alphaI = .8; % learning rate

dataInput = rand(2,4000); % each COLUMN is a data point
% dummy data
% dataInput = [rand(2,2000) 1 + rand(2,100)]; % each column is a data point

dimDataInput = size(dataInput,1); % gives the dimensionality of data space
latticeCell = createInitLattice(dimDataInput,latticeSize); % weights initialization

% Perform self organization
finalLattice = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI);

% giving the final weights of the lattice in Cell form
finalLatticeCell = mat2cell(finalLattice,[ones(1,latticeSize(1))],[ones(1,latticeSize(2))],2);
finalLatticeCell = cellfun(@(x)reshape(x,2,1),finalLatticeCell,'un',0);

% % Plot the mapping and input data
% figure;
% plot(dataInput(1,:),dataInput(2,:),'g.'); hold on; plot(finalLattice(:,:,1),finalLattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot(finalLattice(:,:,1),finalLattice(:,:,2),'k-'); plot(finalLattice(:,:,1)',finalLattice(:,:,2)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); title('Self organised clusters')

densityLattice = calcDensityLattice(finalLattice,dataInput,size(latticeCell));
densityLattice = mat2gray(densityLattice);
figure; imagesc(densityLattice); colormap(flipud(gray)); colorbar

end


function finalLattice = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI)
% the self organizing map steps here

% convert the input lattice cell into a multi-dimensional Matrix 
Z = cellfun(@(x)reshape(x,1,1,[]),latticeCell,'un',0);
lattice = cell2mat(Z); % this is a multi-dimensional Matrix, with third dimension holding different input dimensions

r = (1:size(lattice,1))';c = 1:size(lattice,2); 
latticeIndices(:,:,1) = r(:,ones(1,size(lattice,2))); latticeIndices(:,:,2) = c(ones(1,size(lattice,1)),:);  % latticeIndices : holds the i,j indices of the 2d lattice space

figure;
subplot(2,2,1);
plot(dataInput(1,:),dataInput(2,:),'g.'); hold on; plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
xlabel('First data dimension'); ylabel('Second data dimension'); title('Plot of prototypes in input space : Initial')
legend('Input data vectors','Prototype vectors')
dum = 2;

for i = 1:numIters
%     radius = initRadius; % can do decay here
    radius = initRadius * ((i <= numIters/5) + .8 * (i > numIters/5 & i <= numIters/2) + .5 * (i > numIters/2 & i <= numIters*.8)+ .2 * (i > numIters*.8));
    alpha = alphaI * ((i <= numIters/10) + .5 * (i > numIters/10 & i <= numIters/2.5) + .125 * (i > numIters/2.5 & i <= numIters*.8)+ .025 * (i > numIters*.8));
      
    % pick an x (data point) randomly
    x = dataInput(:,randi(size(dataInput,2)));
   
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = reshape(x,1,1,[]) - lattice; % a 3D matrix of difference between every weight and x
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % finding norm or eucledian distance
  
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner); 
    c = [win_row win_col];
    
    % make a neighbourhood function in a matrix
    neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius);
    
    % update the weights - Learning rule
    lattice = lattice + alpha * neighbourhoodFn .* differenceMatrix;
    
    if sum(i == [numIters/10 numIters/2 numIters])
        % Plot the mapping and input data
        subplot(2,2,dum);
        plot(dataInput(1,:),dataInput(2,:),'g.'); hold on; plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
        plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
        xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
        legend('Input data vectors','Prototype vectors')
        dum = dum + 1;
    end
end
finalLattice = lattice;

end


function densityLattice = calcDensityLattice(lattice,dataInput,sizeOflatticeCell)

densityLattice = zeros(sizeOflatticeCell);

for i = 1:size(dataInput,2)
    x = dataInput(:,i);
    
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = reshape(x,1,1,[]) - lattice; % a 3D matrix
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % a 2D matrix for euclidian distances to x
    
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner); 
    c = [win_row win_col];
    % update the density lattice
    densityLattice(c(1),c(2)) = densityLattice(c(1),c(2)) + 1;
end
    
end


function neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius)

distNeighbour = sum(abs(latticeIndices - reshape(c,1,1,[])),3); % Manhattan distance metric for the neighbourhood function
% EqDistNeighbour = sqrt(sum((latticeIndices - reshape(c,1,1,[])).^2,3)); % eucleidian distance metric for the neighbourhood function
neighbourhoodFn = exp(-((distNeighbour)./(radius)).^2);

end


function latticeCell = createInitLattice(dimDataInput,latticeSize)
% creates random weight vectors in a cell 
latticeCell = cell(latticeSize);
latticeCell = arrayfun(@(x) rand(dimDataInput,1),latticeCell, 'uni',0);
end


% % Extraas : Cell function formulations..
%     % find euclidian distances
%     differenceMatrix = cellfun(@(t) t - x, lattice);
%     distToXMatrix = cellfun(@(t) norm(t - x), lattice);
% 
% distNeighbour = @(w,c) sum(abs(w - c)); % Manhattan distance metric for the neighbourhood function
% %     distNeighbour = @(w,c) norm(w,c); % eucleidian distance metric for the neighbourhood function
% neighbourhoodFn = @(w,c) exp(-((distNeighbour(w,c))/(2*radius))^2);
% 
% Initializations
%     differenceMatrix = zeros(size(lattice)); % a 3D matrix
%     distToXMatrix =  zeros(size(latticeCell)); % a 2D matrix of eucledian distances
%     
% long for loop for calculation of eucliean distances         
%     for k = 1:size(lattice,2)
%         for j = 1:size(lattice,1)
% %             differenceMatrix(j,k,:) = x - reshape(lattice(j,k,:),size(x));
%             distToXMatrix(j,k) = norm(reshape(differenceMatrix(j,k,:),size(x)));
%         end
%     end
% 
% long for loop for finding distance and making neighbourhood functions
% for j = 1:size(lattice,2)
%     for i = 1:size(lattice,1)
%         w = [i j];
% %         distNeighbour = sum(abs(w - c)); % Manhattan distance metric for the neighbourhood function
%         %     distNeighbour = norm(w,c); % eucleidian distance metric for the neighbourhood function
%         neighbourhoodFn(i,j,:) = exp(-((distNeighbour)/(radius))^2);
