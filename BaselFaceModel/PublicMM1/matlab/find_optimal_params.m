function total_error = find_optimal_params(shapeMU, shapePC, shapeEV, texMU,   texPC, tl,  texEV, segMM, segMB, myrp, target)

% Start with random weights
alpha = randn(2, 4);   % Random shape   coefficients of 4 segments
beta  = randn(2, 4);   % Random texture coefficients of 4 segments
 
% Parameters
sigma_N = 4000

% Apply initial parameters
new_shape  = coef2object( alpha, shapeMU, shapePC, shapeEV, segMM, segMB );
save_shape = new_shape;
orig_beta = beta;
new_texture    = coef2object( beta,  texMU,   texPC,   texEV,   segMM, segMB );
source = display_face(new_shape, new_texture, tl, myrp);

% Compute the cost function
total_diff = 0;

% Learning rate
%lambda = 0.03;
lambda = 0.5;
size_src = size(source);

count = 0;

a = size_src(1);
b = size_src(2);
c = a*b;
my_error = zeros(c,1);

for i=1:size_src(1)
    for j = 1:size_src(2)
        count = count + 1;
        R_src = source(i, j, 1);
        G_src = source(i, j, 2);
        B_src = source(i, j, 3);
        
        R_tgt = target(i, j, 1);
        G_tgt = target(i, j, 2);
        B_tgt = target(i, j, 3);
        
        my_error(count) =  abs(R_src - R_tgt) + abs(G_src - G_tgt)+ abs(B_src - B_tgt);
        
    end
end

total_error = sum(my_error);



% Learn alpha, beta parameters using gradient decent
num_iters = 5000;

alpha_sz = size(alpha);
gradient = zeros(alpha_sz(1), alpha_sz(2));
gradient_beta = zeros(alpha_sz(1), alpha_sz(2));

sigma_T = 4000;

for i = 1:1000
    i
    total_error
    for j = 1:2
        for k = 1:4
            % Learn alpha
            noise_factor = 1 + (rand - 0.5)*0.1;
            
            gradient(j) = (-2/(sigma_N * sigma_N)) * (total_error) + 2 * alpha(j,k)/(shapeEV(j) * shapeEV(j));
            alpha(j,k) = alpha(j,k) - lambda * gradient(j) * noise_factor;
            new_shape  = coef2object( alpha, shapeMU, shapePC, shapeEV, segMM, segMB );
            
            a = size_src(1);
            b = size_src(2);    
            c = a*b;
            my_error = zeros(c,1);
            
            % Learn beta
            gradient_beta(j) = (-2/(sigma_T * sigma_T)) * (total_error) + 2 * beta(j,k)/(texEV(j) * texEV(j));
            beta(j,k) = beta(j,k) - lambda * gradient_beta(j) * noise_factor;
            new_texture  = coef2object(beta, texMU, texPC, texEV, segMM, segMB );
  
            source = display_face(new_shape, new_texture, tl, myrp);
            
            for m=1:size_src(1)
                for n = 1:size_src(2)
                    count = count + 1;
                    R_src = source(m, n, 1);
                    G_src = source(m, n, 2);
                    B_src = source(m, n, 3);

                    R_tgt = target(m, n, 1);
                    G_tgt = target(m, n, 2);
                    B_tgt = target(m, n, 3);

                    my_error(count) =  abs(R_src - R_tgt) + abs(G_src - G_tgt)+ abs(B_src - B_tgt);
                end
            end
            total_error = sum(my_error);
            gradient(j);
        end
    end
end
