function [ Em ] = Elimination_mat( m )
%m Size of E
T = tril(ones(m)); % Lower triangle of 1's
f = find(T(:)); % Get linear indexes of 1's
k = m*(m+1)/2; % Row size of L
m2 = m*m; % Colunm size of L
L = zeros(m2,k); % Start with L'
x = f + m2*(0:k-1)'; % Linear indexes of the 1's within L'
L(x) = 1; % Put the 1's in place
Em = L'; % Now transpose to actual L


end

