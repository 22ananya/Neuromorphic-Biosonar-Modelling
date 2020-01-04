function [b,a] = FOD_GF(fc, n, fs, bw)
b   = zeros(1,n+1);
a   = zeros(1,2*n+1);

% convert to radians
theta   = 2*pi*fc/fs;
phi     = 2*pi*bw/fs;
alpha   = -exp(-phi)*cos(theta);
b1      = 2*alpha;
b2      = exp(-2*phi);
a0      = abs((1+b1*cos(theta)-1i*b1*sin(theta)+b2*cos(2*theta)-1i*b2*sin(2*theta)) / (1+alpha*cos(theta)-1i*alpha*sin(theta)));

% Compute the position of the pole
atilde  = exp(-phi-1i*theta);

% Repeat the conjugate pair n times, and expand the polynomial
a2      = poly([atilde*ones(1,n),conj(atilde)*ones(1,n)]);

% Compute the position of the zero, just the real value of the pole
btilde  = real(atilde);

% Repeat the zero n times, and expand the polynomial
b2  = poly(btilde*ones(1,n));

% Amplitude scaling
b2  = b2*(a0^n);    % Scale to get 0 dB attenuation

% Place the result (a row vector) in the output matrices.
b(1,:) = b2;
a(1,:) = a2;
end       