% hw6 Problem 2
function SOM2
% Prashant Kalvapalle 
% Comp 504 HW6 - Base code for all problems
% close 1;  
% NOTE : Initial (and final) lattice is a cell representation, In the function it is
% used as a multi-dimensional matrix

latticeSize = [8 8]; 
initRadius = max(latticeSize); % Initial radius of influence

numIters = 20000; % number of learning steps
alphaI = .8; % learning rate

% Input data entry
dataInput = [createGaussians([2 1000],.1,[7 7]), createGaussians([2 1000],.1,[0 7]), createGaussians([2 1000],.1,[7 0]), createGaussians([2 1000],.1,[0 0]),]; % each COLUMN is a data point

dimDataInput = size(dataInput,1); % gives the dimensionality of data space
latticeCell = createInitLattice(dimDataInput,latticeSize); % weights initialization

% Perform self organization
[finalLattice, stepsToConv] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI);

% % giving the final weights of the lattice in Cell form
% finalLatticeCell = mat2cell(finalLattice,ones(1,latticeSize(1)),ones(1,latticeSize(2)),2); finalLatticeCell = cellfun(@(x)reshape(x,2,1),finalLatticeCell,'un',0);

[densityLattice, ~, histoData] = calcDensityLattice(finalLattice,dataInput,size(latticeCell));
densityLattice = mat2gray(densityLattice);
figure; imagesc(densityLattice); colormap(flipud(gray)); colorbar; title('Density of Inputs mapped to each Prototype')

if stepsToConv < numIters
    disp(['SOM Converged in ',num2str(stepsToConv),' steps'])
else 
    disp(['Maximum iterations exhausted = ',num2str(stepsToConv),' steps'])
end

plotHistoChart(histoData);

end


function [finalLattice, stepsToConv] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI)
% the self organizing map steps here

% convert the input lattice cell into a multi-dimensional Matrix 
Z = cellfun(@(x)reshape(x,1,1,[]),latticeCell,'un',0);
lattice = cell2mat(Z); % this is a multi-dimensional Matrix, with third dimension holding different input dimensions

r = (1:size(lattice,1))';c = 1:size(lattice,2); 
latticeIndices(:,:,1) = r(:,ones(1,size(lattice,2))); latticeIndices(:,:,2) = c(ones(1,size(lattice,1)),:);  % latticeIndices : holds the i,j indices of the 2d lattice space

figure(1);
subplot(2,2,1);
dI = reshape(dataInput',[],4,2);
plot(dI(:,:,1),dI(:,:,2),'.'); hold on; plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
xlabel('First data dimension'); ylabel('Second data dimension'); title('Plot of prototypes in input space : Initial')
legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')

dum = 2;
% [~, oldMapData, ~] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
stepsToConv = numIters;

for i = 1:numIters
%     radius = initRadius; % can do decay here
decayIters = 10000;
radius = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
alpha = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
      
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
    
    % Checking for convergence every 1000 steps
%     if mod(i,1000) == 0
% %         mapData = calcDataMapping(lattice,dataInput); 
%         [~, mapData, histoData] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
%         match = (mapData(1,:) == oldMapData(1,:) & mapData(2,:) == oldMapData(2,:));
% %         figure(3); hold on; plot(i,sum(match),'k.');
%         if sum(match)/size(dataInput,2) >= (1 - 1e-3) % < .1 percentage change in prototype assignment
%             stepsToConv = i;
%         else
%             oldMapData = mapData;
%         end
%     end
    % making plots at particular learning steps as defined in the vector
    if sum(i == [decayIters/10 decayIters/2 decayIters])
        % Plot the mapping and input data
        figure(1); subplot(2,2,dum);
        dI = reshape(dataInput',[],4,2);
        plot(dI(:,:,1),dI(:,:,2),'.'); hold on; plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
        plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
        xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
        legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
        dum = dum + 1;
    end
%     % making plots every 1000 learning steps to visually approximate
%     % learning steps to convergence
%         if ~mod(i,1000)
% %         Plot the mapping and input data
%         figure(2)
% %         subplot(2,2,dum);
%         dI = reshape(dataInput',[],4,2);
%         plot(dI(:,:,1),dI(:,:,2),'.'); hold on; plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
%         plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
%         xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
%         legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
%         hold off; 
%         drawnow; 
% %         dum = dum + 1;
%         end
    
    if stepsToConv < numIters
        break
    end
    
end
finalLattice = lattice;

end


function [densityLattice,mapData,histoData]  = calcDensityLattice(lattice,dataInput,sizeOflatticeCell)
% calculates the prototype each data point is mapped to
densityLattice = zeros(sizeOflatticeCell);
mapData = zeros([2 size(dataInput,2)]);
histoData = zeros([sizeOflatticeCell,4]);
% seq = [ones(1,1000) 2*ones(1,1000) 3*ones(1,1000) 4*ones(1,1000)];
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
    mapData(:,i) = [win_row win_col];
    histoWrite = ([1 0 0 0] * (i <= 1000) + [0 1 0 0] * (i > 1000 & i <= 2000) + [0 0 1 0] * (i > 2000 & i <= 3000) + [0 0 0 1] * (i > 3000 & i <= 4000));
    histoData(c(1),c(2),:) = histoData(c(1),c(2),:) + (reshape(histoWrite,1,1,[]));
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


function x1 = createGaussians(dim,var,mean)
% creates a normal random vector with dim dimension. mean of first 2
% dimensions set by [mean1 mean2]
x1 = sqrt(var)*randn(dim(1),dim(2));
% x1=detrend(x1);
x1(1,:) = x1(1,:) + mean(1);
x1(2,:) = x1(2,:) + mean(2);
end


function plotHistoChart(histoData)
figure;
m = size(histoData,1); n = size(histoData,2);
p = 1;
for j = 1:n
    for i = 1:m
        ax = axes('position',[(i-1)/m (n-j)/n 1/m 1/n]); hold on;
        hiss = reshape(histoData(i,j,:),1,4,[]);
        colors = {'r', 'b', 'g', 'y'};
        % Plots different bars for each data type
        for k = 1:numel(hiss)
            bar(ax,k, hiss(k),colors{k});
        end
        ylim([0 200]);
        set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]);
        set(ax,'Box','on')
%         set(gca,'Visible','off');
%         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
        p = p + 1;

    end
end
end


%
%
% % Plot the mapping and input data
% figure;
% plot(dataInput(1,:),dataInput(2,:),'g.'); hold on; plot(finalLattice(:,:,1),finalLattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot(finalLattice(:,:,1),finalLattice(:,:,2),'k-'); plot(finalLattice(:,:,1)',finalLattice(:,:,2)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); title('Self organised clusters')
% 
% Old function - built into the calcDensityLattice function
% function mapData = calcDataMapping(lattice,dataInput)
% % outputs a vector showing the prototype location where each data point maps 
% mapData = zeros([2 size(dataInput,2)]);
% 
% for i = 1:size(dataInput,2)
%     x = dataInput(:,i);
%     
%     % find euclidian distances and difference between chosen x and all W's
%     differenceMatrix = reshape(x,1,1,[]) - lattice; % a 3D matrix
%     distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % a 2D matrix for euclidian distances to x
%     
%     % find the winner = c = [win_row win_col]
%     [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner); 
%     % update the density lattice
%     mapData(:,i) = [win_row win_col];
% end
%     
% end
%           Extra commands in the bar plot
%         a = subplot(m,n,p);
%         a = axes('parent', h);
%         hold(a, 'on')
%         
%         colors = {'r', 'b', 'g', 'y'};
%         somenames = {'IND Relation'; 'DIS Relation'; 'EQ Relation'};
%         
%         for k = 1:numel(hiss)
%             b = bar(k, x(k), 0.1, 'stacked', 'parent', a, 'facecolor', colors{i});
%         end
%         
%         a.XTick = 1:3;
%         a.XTickLabel = somenames;
%         
%         ylabel('F1')
% ------------------
%         subplot(m,n,p); hold on;
%         hiss = reshape(histoData(i,j,:),1,4,[]);
%         colors = {'r', 'b', 'g', 'y'};
%         % Plots different bars for each data type
%         for k = 1:numel(hiss)
%             bar(k, hiss(k),colors{k});
%         end
%         ylim([0 200]);
%         set(ax2,'YTickLabel',[]);
%         set(gca,'Visible','off');
% %         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
%         p = p + 1;