function imgData = reconstructImage(vecSet)
% recustructs image from the 64x1 vectors of the 8x8 blocks
% Inputs a cell of the 64x1 vectors
% Outputs the reconstructed image

m = 192; n = 256; % give the orignal image dimensions
% m = 32; n = 32; % give the orignal image dimensions
C = cellfun(@(x)reshape(x,8,8),vecSet,'un',0); % reform blocks
C = reshape(C,m/8,n/8); % align blocks where they belong in the 2x2 cell 
imgData = cell2mat(C);
end