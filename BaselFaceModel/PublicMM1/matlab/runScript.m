
myrp = defrp;
load('01_MorphableModel.mat');
target = imread('target.png');
find_optimal_params(shapeMU, shapePC, shapeEV, texMU,texPC, tl,texEV, segMM, segMB,myrp, target);