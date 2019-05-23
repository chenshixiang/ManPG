function [ x_prox,Act_set ,Inact_set] = prox_l1(  b ,lambda,r )

    a = abs(b)-lambda;
    if r < 15
      Act_set = double( a > 0);
    else
      Act_set = ( a > 0);
    end
    x_prox = (Act_set.*sign(b)).*a;
    if nargout==3
         Inact_set= (a <= 0);
    end

end

