function result = sigmoid(z)

result = zeros(size(z));
result = 1 ./ (1 .+ e.^(-z));

end