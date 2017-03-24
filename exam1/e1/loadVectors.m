function data = loadVectors(ocelot)
% breaks image into 8x8 blocks
% Outputs a cell with 64x1 vectors in it
m = size(ocelot,1); n = size(ocelot,2);
a = repmat(8,1,m/8); b = repmat(8,1,n/8);

C = mat2cell(ocelot,a,b); % break into blocks
C = reshape(C,numel(C),1); % align all blocks in 1 column
data = cellfun(@(x)reshape(x,64,1),C,'un',0);

end