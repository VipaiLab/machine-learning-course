function table = link_regulation(x,y)
    table = zeros(x,y);
    max_ = 2^x - 1;
    for k = 1 : y
        temp = mod(k,max_);
        temp = (temp == 0)*max_ + temp;
        temp_bin = dec2bin(temp,x);
        table(:,k) = temp_bin;
    end
    table = table - 48;
end