function R = corrmat_DCGP(x1,x2,omega)
    n = size(x1,1);     m = size(x2,1);
	omega = sqrt(10.^omega);
	x1 = x1.*repmat(omega, n, 1);
	x2 = x2.*repmat(omega, m, 1);
    Rx = zeros(n,m);
	if n >= m      
		for i = 1:m 
           Rx(:, i) = sum((x1 - repmat(x2(i, :), n, 1)).^2, 2); 
        end
	else
		for i = 1:n
           Rx(i, :) = sum((repmat(x1(i, :), m, 1) - x2).^2, 2); 
        end		
    end
	R = exp(-Rx);
end