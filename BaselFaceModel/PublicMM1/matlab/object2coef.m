function coef = object2coef(obj, mu, pc, ev, segbin)
%OBJECT2COEF projects a shape or texture to the MM yielding its coefficients.
%   COEF = OBJECT2COEF(OBJ, MU, PC, EV) projects a shape or texture vector OBJ to the MM
%   given by the mean MU, the principal components PC and their strandard deviation EV.
%
%   COEF = OBJECT2COEF(OBJ, MU, PC, EV, SEGBIN) performs a segmented projection. SEGBIN
%   is a matrix that specifies which vertex belongs to a segment. The number of segment
%   is equal to the number of columns of SEGBIN, and its number of rows is equal to the
%   number of vertex in the MM. Each column of SEGBIN is a binary vector (vertex with 0 do
%   not belong to the segment). The number of columns of COEF is equal to the number of
%   segments. Note that the SAME model is used for all segments (a PCA was not performed
%   for each segment). Hence, the projection requires four matrix inversion.
%
%------------- BEGIN CODE --------------

% Arguments checking
if nargin ~= 4 && nargin ~= 5
  error('Inappropriate number of arguments')
end

if nargout ~= 1
  error('One output argument required')
end

if nargin > 4
  n_seg = size(segbin, 2);
else
  n_seg = 1;
end

obj = obj - mu;
if n_seg == 1
  % Projection to PCA space
  coef = (pc' * obj) ./ ev;
else
  % Projection to a segmented PCA space
  %coef = zeros( size(pc,2), n_seg, class(pc) );
  
  
  for i=1:n_seg
    idx = idx2intl( find(segbin(:,i)), 3 );
    coef(:, i) = (pc( idx, : ) \ obj(idx)) ./ ev;
  end
end

%------------- END OF CODE --------------
