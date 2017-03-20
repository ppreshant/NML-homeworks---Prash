%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function to demo the use of LVQ1 for classification of the three
% regions shown here by the matrix of class labels for the respective
% spatial locations
%
% Adapted from Ham and Kostanic, 2000
% for course COMP / ELEC / STAT 502, E. Merenyi
%
%   A = [3 1 1 1 1 2 3 3 3
%   	 3 3 3 3 3 2 3 3 3
%   	 3 3 3 3 3 1 2 2 2
%   	 3 1 1 1 1 1 1 1 1
%   	 3 1 1 1 2 1 1 1 1
%   	 3 1 1 2 2 2 1 1 1
%   	 3 1 2 2 1 3 2 2 2
%   	 3 3 3 3 3 3 2 2 2
%   	 3 3 3 3 3 3 3 3 2];
%%
function LVQ_prash

% clear all;
close all;

nrows = 9; ncols = 9; ndim = 2; % Set the dimensions of the input image
N = nrows*ncols;  % We will unfold the input image into a vector of length N
nP = 60;          % Number of LVQ prototypes (weight vectors)
nC = 3;           % Number of classes
%mu = 0.05;       % Learning rate -- NOT taken from here. Set in the decay
% schedule below
maxsteps = 100000; % Max. number of learning steps allowed. Increase this as needed.
%mfr = 2000;      % Monitoring frequency --- NOT taken from here. Set in
% the decay schedule below
% Decay schedule for learning rate mu, in the form as shown
% LRsched = [learn-rate1 learn-rate2 ... learn-ratek
%           tp-learn-step1 to-learn-step2 ... to-learn-stepk ] % use
%           learn-rate1 up to learn-step1, etc.
LRsched =  [0.5 0.2 0.1 0.005
    10000 20000 120000 inf ]; % Decay schedule for learning rate mu
Mfrsched = [200 500 5000 20000
    1000 5000 50000 inf ]; % Schedule of monitoring frequency
[lr1,lr2] = size(LRsched);  % Get the size of the decay step function
[mf1,mf2] = size(Mfrsched); % Get the sze of the monitoring function

% disp 'N, np, nC, mu, maxsteps'
% N
% nP
% nC
maxsteps

% Generate the training data:
% Specify the matrix of class labels for the spatial locations
A = [ 3 1 1 1 1 2 3 3 3
    3 3 3 3 3 2 3 3 3
    3 3 3 3 3 1 2 2 2
    3 1 1 1 1 1 1 1 1
    3 1 1 1 2 1 1 1 1
    3 1 1 2 2 2 1 1 1
    3 1 2 2 1 3 2 2 2
    3 3 3 3 3 3 2 2 2
    3 3 3 3 3 3 3 3 2];



% Assign class labels to prototypes
%    Cw = randsample(1:nC,nP,true); % To assign labels randomly
% Here I assign labels by hand, divide them evenly among prototypes
nLperClass = ceil(nP/nC); % This may not devide evenly into nP, take care
% The label assignment below is hard wired for the problem above, i.e., for 3
% classes. Generalize / extend for more classes.
Cw = zeros(1,nP);
Cw(1:nLperClass)=1;
Cw(nLperClass+1:2*nLperClass)=2;
Cw(2*nLperClass+1:nP)=3;

% Set a color table to represent class labels on plots
% This is for eight classes, you can generate larger ones and
% store / read in from a file, make it a parameter, etc.
%r  or  g   y   w   b    p gray
label_color_map_r = [255;254;  0;252;255; 47;164;180];
label_color_map_g = [  0;146;255;228;255;146;  0;180];
label_color_map_b = [  0; 43;  0; 98;255;255;175;180];
label_color_map   = [label_color_map_r label_color_map_g label_color_map_b]/255;

disp 'Target classification, A'
% figure, imagesc(A), colormap('gray'); % Check the image of target class labels
figure, imagesc(A), colormap(label_color_map([1 5 7],:)); % use colors for classes
title('Target classification')

F(17) = struct('cdata',[],'colormap',[]); % allocate frames for saving figures

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% No change should be needed below this line for learning
%%%%%%% the classification of a different image, with different
%%%%%%% number of prototyopes.

% Set the input vectors and associated categories
% We will assign the spatial coordinates of the labels in A
% to the ndim elements of each of the N, in a row-wise fashion
% and center the coordinates if desired
%
X = zeros(ndim,N);
Cx = zeros(1,N);
k = 0;
for i = 1:nrows
    for j = 1:ncols
        k = k+1;
        %       X(1,k) = i-ceil(nrows/2); % to center the y coordinates if desired
        %      	X(2,k) = j-ceil(ncols/2); % to center the x coordinates if desired
        X(1,k) = i;
        X(2,k) = j;
        %      	Cx(k) = A(ncols-j+1, i); % this seems to flip
        % then requires a flipud at the end
        Cx(k) = A(j,i);
    end
end

% Learning LVQ1 function
[W,Cw,lstep,errorHistory] = LVQ1_learn(A,X,Cw,Cx,ndim,N,nP,label_color_map,LRsched,Mfrsched,lr2,mf2,maxsteps);

% LVQ1 recall function
[predicted_classes,errorCount] = recall(X,W,Cw,N,nP,label_color_map,nrows,ncols,lstep,Cx);

disp(strcat('Training data : error rate = ',num2str(errorCount),'/81'))

% Insert here generation of test data
[Atest,X_test,Cx_test,N_test,nrows_test,ncols_test] = generateTestData(A);

% target classification for test data
figure, imagesc(Atest), colormap(label_color_map([1 5 7],:)); % use colors for classes
title('Test Data : Target classification')

% Recall on test data
[predicted_classes_test,test_errorCount] = testRecall(X_test,W,Cw,N_test,nP,label_color_map,nrows_test,ncols_test,Cx_test);
disp(strcat('Test data : error rate = ',num2str(test_errorCount),'/324'))
% plotting learning history
figure; plot(errorHistory(2,:),errorHistory(1,:)); xlabel('Learning steps'); ylabel('Number of misclassifications'); 
title('Learning History')

end



function [W,Cw,lstep,errorHistory] = LVQ1_learn(A,X,Cw,Cx,ndim,N,nP,label_color_map,LRsched,Mfrsched,lr2,mf2,maxsteps)
%% Implement the LVQ1 learning here to see how all of this hangs together
% Normally, this should be in a function and called.
%
%%%%%%%%%%%%%% begin LVQ1 learning %%%%%%%%%%%%%%%%
%
% Initialize prototypes
%
data_min=min(min(X));
data_max=max(max(X));
% Initialize the weights in the range of the input data
scale=1; % Make this a parameter if you like
W=scale*(rand(ndim,nP)*(data_max - data_min)+data_min);
%
% % Plot initial state of prototypes on target data
% %     figure, imagesc(A), colormap('gray'); hold on
% figure, imagesc(A), colormap(label_color_map([1 5 7],:)); hold on
% plot(W(1,Cw==1),W(2,Cw==1),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(1,:),'MarkerSize',10)
% plot(W(1,Cw==2),W(2,Cw==2),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(5,:),'MarkerSize',10)
% plot(W(1,Cw==3),W(2,Cw==3),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(7,:),'MarkerSize',10)
% % plot(W(1,:),W(2,:),'o','Color', label_color_map(Cw(:),(:))
% % plot(W(1,:),W(2,:),'o','Color', label_color_map(Cw(:),(:))
% title('Target classes and prototype positions, learn step = 0');
% curFrame = 1;
% F(curFrame) = getframe(gcf); curFrame=curFrame+1;
%%

mu = LRsched(1,1); % initial value of the learning rate
mfr = Mfrsched(1,1); % initial value of the monitoring rate

dum = 1; errorHistory = zeros(2,20);
errorHistory(1,dum) = errorRecall(X,W,Cw,Cx,N,nP); 
errorHistory(2,dum) = 1; dum = dum + 1;
% Insert here setup for arrays to collect errors for learning history
% and to use as stopping criteria

for lstep = 1:maxsteps
    i = randsample(1:N,1); % Select sample index randomly
    %
    d = zeros(1,nP);
    for j = 1:nP
        d(j) = norm(W(:,j)-X(:,i));
    end
    [mindist,I] = min(d);
    if Cw(I) == Cx(i)
        W(:,I) = W(:,I) + mu * (X(:,i)-W(:,I));
    else
        W(:,I) = W(:,I) - mu * (X(:,i)-W(:,I));
    end
    %
    if lstep == maxsteps
        lstep%, mu, mfr
        % plotting of prototypes in data space if ndim <3
        if ndim < 3
            %      figure, imagesc(A), colormap('gray'); hold on
            figure, imagesc(A), colormap(label_color_map([1 5 7],:)); hold on
            plot(W(1,Cw==1),W(2,Cw==1),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(1,:),'MarkerSize',10)
            plot(W(1,Cw==2),W(2,Cw==2),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(5,:),'MarkerSize',10)
            plot(W(1,Cw==3),W(2,Cw==3),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(7,:),'MarkerSize',10)
            title(strcat('Target classes and prototype positions, learn step = ',num2str(lstep)));
%             F(curFrame) = getframe(gcf); curFrame=curFrame+1;
        end      
        
        % Check for stopping conditions.
        %
        % Since I am plotting the prototypes in the data space here with
        % the monitoring frequency mfr I keep mfr low. It may be practical
        % to set up a different
        % monitoring frequency for collecting errors for learning history.
        %
    end
    
    % Insert here recall for training and test data, and
    % computation of classification errors, recording errors for
    % learning history.
    if ~mod(lstep,mfr)
        errorHistory(1,dum) = errorRecall(X,W,Cw,Cx,N,nP);
        errorHistory(2,dum) = lstep; dum = dum + 1;
    end
    
    % Decrease learn rate and monitoring frequency according to schedules
    for lr = 2:lr2
        if lstep > LRsched(2,lr-1)
            mu = LRsched(1,lr);
        end
    end
    for fr = 2:mf2
        if lstep > Mfrsched(2,fr-1)
            mfr = Mfrsched(1,fr);
        end
    end
    
end
%%%%%%%%%%%%%% end LVQ1 learning %%%%%%%%%%%%%%%%
end


function [predicted_classes,errorCount] = recall(X,W,Cw,N,nP,label_color_map,nrows,ncols,lstep,Cx)
%%
% Test the classification of the training data using the prototypes
% (weights, W) learned
Cxhat = zeros(1,N);
for i = 1:N
    d = zeros(1,nP);
    for j = 1:nP
        d(j) = norm(W(:,j)-X(:,i));
    end
    [~,I] = min(d);
    Cxhat(i) = Cw(I);
end
%
% Insert here recall for test data (or make the above recall into a function and
% call it with training and with test data)

% Reshape to original spatial image format, and display
Cxhat_reshaped = reshape(Cxhat,nrows,ncols);
% predicted_classes = flipud(Cxhat_reshaped);
predicted_classes = Cxhat_reshaped;
%    disp 'End';
%    imagesc(Cxhat_reshaped); colormap('gray');
%
% Display predicted classes for training data, and superimpose the LVQ prototypes
figure, imagesc(predicted_classes), colormap(label_color_map([1 5 7],:)); hold on
plot(W(1,Cw==1),W(2,Cw==1),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(1,:),'MarkerSize',10)
plot(W(1,Cw==2),W(2,Cw==2),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(5,:),'MarkerSize',10)
plot(W(1,Cw==3),W(2,Cw==3),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(7,:),'MarkerSize',10)
title(strcat('Predicted classes for training data, and prototype positions, learn step = ',num2str(lstep)));
%      F(curFrame) = getframe(gcf); curFrame=curFrame+1;
%       end
%
% Insert display of predicted classes for test data, and superimpose the LVQ prototypes

errorCount = sum(Cxhat ~= Cx);
end


function [predicted_classes,errorCount] = testRecall(X,W,Cw,N,nP,label_color_map,nrows,ncols,Cx)
%%
% Test the classification of the training data using the prototypes
% (weights, W) learned
Cxhat = zeros(1,N);
for i = 1:N
    d = zeros(1,nP);
    for j = 1:nP
        d(j) = norm(W(:,j)-X(:,i));
    end
    [~,I] = min(d);
    Cxhat(i) = Cw(I);
end
%
% Insert here recall for test data (or make the above recall into a function and
% call it with training and with test data)

% Reshape to original spatial image format, and display
Cxhat_reshaped = reshape(Cxhat,nrows,ncols);
% predicted_classes = flipud(Cxhat_reshaped);
predicted_classes = Cxhat_reshaped;
%    disp 'End';
%    imagesc(Cxhat_reshaped); colormap('gray');
%
 W = 2 * W;
% Display predicted classes for training data, and superimpose the LVQ prototypes
figure, imagesc(predicted_classes), colormap(label_color_map([1 5 7],:)); hold on
plot(W(1,Cw==1),W(2,Cw==1),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(1,:),'MarkerSize',10)
plot(W(1,Cw==2),W(2,Cw==2),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(5,:),'MarkerSize',10)
plot(W(1,Cw==3),W(2,Cw==3),'o','LineWidth',2.0,'MarkerEdgeColor','k','MarkerFaceColor', label_color_map(7,:),'MarkerSize',10)
title(strcat('Predicted classes for testing data, and prototypes'));
%      F(curFrame) = getframe(gcf); curFrame=curFrame+1;
%       end
%
% Insert display of predicted classes for test data, and superimpose the LVQ prototypes
errorCount = sum(Cxhat ~= Cx);

end


function errorCount = errorRecall(X,W,Cw,Cx,N,nP)
Cxhat = zeros(1,N);
for i = 1:N
    d = zeros(1,nP);
    for j = 1:nP
        d(j) = norm(W(:,j)-X(:,i));
    end
    [~,I] = min(d);
    Cxhat(i) = Cw(I);
end

errorCount = sum(Cxhat ~= Cx);
end


function [Atest,X,Cx,N,nrows,ncols] = generateTestData(A)
Atest = zeros(18,18);
Atest(1:2:17,1:2:17) = A; Atest(2:2:18,1:2:17) = A; Atest(1:2:17,2:2:18) = A; Atest(2:2:18,2:2:18) = A;

N = numel(Atest); nrows = size(Atest,1); ncols = size(Atest,2); ndim = 2; % Set the dimensions of the input image

X = zeros(ndim,N);
Cx = zeros(1,N);
k = 0;

for i = 1:nrows
    for j = 1:ncols
        k = k+1;
        %       X(1,k) = i-ceil(nrows/2); % to center the y coordinates if desired
        %      	X(2,k) = j-ceil(ncols/2); % to center the x coordinates if desired
        X(1,k) = i/2; % since it is a fine grained grid, the step size is 1/2
        X(2,k) = j/2;
        %      	Cx(k) = A(ncols-j+1, i); % this seems to flip
        % then requires a flipud at the end
        Cx(k) = Atest(j,i);
    end
end

end


function makeMovie(F)
%%
% Save figures in a movie
fig = figure; movie(fig,F,1,0.5); % movie(figure_handle,Movie_Frames,Replay_Count,FPS)
%%
% Make and save a video file of this movie
v = VideoWriter('OutPutVideo.avi');
open(v);
for  ii = 1:length(F)
    writeVideo(v,F(ii));
end
close(v);
end