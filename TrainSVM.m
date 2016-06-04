function [w, b] = TrainSVM(Xtrain, ys, lambda, T)

[d, n] = size(Xtrain);

b=0;
w = zeros(d, 1);

epochs = 20;

for i = 1:epochs
    inds = randperm(n);
    for t = 1:T
        nt = 1/(lambda*t);
            
        if ( ys(inds(t)) * (w'* Xtrain(:, inds(t)) + b ) < 1 )
            w = (1 - nt*lambda)*w + nt*ys(inds(t))*Xtrain(:, inds(t));
            b = b + nt*ys(inds(t));
        else
            w = (1 - nt*lambda)*w;
        end
    end
    % Normalize: optional and expensive
    %a = min(1, 1/( norm(w) * sqrt(lambda) ));
    %w = a * w;
    %b = a * b;
end

