function X = positive_CMs(X,r)
%flip the negative columns of CMs to be positive

 for i = 1:r
    [~,I] = max(abs(X(:,i)));
    if X(I,i) < 0 
        X(:,i) = - X(:,i);
    end
 end
end

