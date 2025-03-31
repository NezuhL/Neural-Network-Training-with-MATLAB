rng(12345678)

Number_of_neurons = 10 + mod(randi(100, 1), 10);

angles = randi(360, 300, 1)*pi/180 - pi;
radii  = [5*ones(100,1); 10*ones(100, 1); 12*ones(100, 1)];

[observations(:,1), observations(:,2)]= pol2cart(angles, radii);

figure; plot(observations(1:100, 1), observations(1:100, 2), 'bx')
hold on
plot(observations(101:200, 1), observations(101:200, 2), 'gx')
hold on
plot(observations(201:300, 1), observations(201:300, 2), 'rx')

labels = zeros(3, 300);
labels(1, 1:100)   = 1;
labels(2, 101:200) = 1;
labels(3, 201:300) = 1;

%%%%%%%%%%%%%%%%%%%%%

input_neurons=size(observations, 2);
hidden_neurons=Number_of_neurons;
output_neurons=size(labels, 1);
data=observations;

% Initialize parameters
epochs = 7000;
learning_rate = 0.002;
stopSIGN = 1e-6;
    
% Initialize weights and biases randomly 
W1 = randn(input_neurons, hidden_neurons);
b1 = rand(1, hidden_neurons)*0.0001;
W2 = randn(hidden_neurons, output_neurons);
b2 = rand(1, output_neurons)*0.0001;

% Initialize array to store loss values
loss_values = zeros(epochs, 1);

% Initialize array to store gradients
grad_W1 = zeros(size(W1));
grad_b1 = zeros(size(b1));
grad_W2 = zeros(size(W2));
grad_b2 = zeros(size(b2));

for iter = 1:epochs
    % Forward pass
    a = data * W1 + b1;
    h = Sigmoid(a);  % Applying sigmoid activation
    c = h * W2 + b2;
    q = Softmax(c); %Applying softmax activation
    c = h * W2 + b2;
   
    % Compute loss and gradient 
    N = size(labels, 2); % Number of samples
    K = size(labels, 1); % Number of output neurons
    loss = 0;
    dq = zeros(size(q));
    for i = 1:N
    % Compute loss for each sample
        for k = 1:K
            loss = loss + 0.5 * (q(i, k) - labels(k, i))^2;
        % Compute gradient for each sample
            dq(i, k) = q(i, k) - labels(k, i);
        end
    end
   % Compute average loss
    loss = loss / N;
    loss_values(iter) = loss;


    % Backpropagation
    dc = q.*(1-q).*dq;  
    grad_W2 = h' * dc;
    grad_b2 = sum(dc, 1);
    dh = dc * W2';
    da = dh .* sigmoid_derivative(a); 
    grad_W1 = data' * da;
    grad_b1 = sum(da, 1);

    
    % Parameter updates (gradient descent)
    W1 = W1 - learning_rate * grad_W1;
    b1 = b1 - learning_rate * grad_b1;
    W2 = W2 - learning_rate * grad_W2;
    b2 = b2 - learning_rate * grad_b2;
    
   norms = [norm(grad_W1), norm(grad_W2), norm(grad_b1), norm(grad_b2)];
   gradient_norm = norm(norms);
    
    % Check stopping criterion based on gradient norm
   
    if gradient_norm < stopSIGN
        fprintf('Stopping criterion reached. Gradient norm is below threshold.\n');
        break;
    end

    
end

% Plot the variation of loss over iterations
figure;
plot(1:epochs, loss_values, 'b-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Loss');
title('Loss over Iterations');
grid on;



%FUNCTIONS:

% Sigmoid activation function
function y = Sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end


% Softmax activation function
function y = Softmax(x)
    exp_x = exp(x);
    y = exp_x ./ sum(exp_x, 2);
end


% Sigmoid activation function derivative
function y = sigmoid_derivative(x)
    y = Sigmoid(x) .* (1 - Sigmoid(x));
end

