% Load data using readNPY if using NPY format
X = readNPY('data/u_train.npy'); % Input data
Y = readNPY('data/output_train.npy'); % Output data

% Get the number of samples in your dataset
numSamples = length(Y);

% Define the split ratio for training (e.g., 70% for training)
splitRatio = 0.7;
numTrain = round(splitRatio * numSamples);

% Split input data (X) and output data (Y) into training and testing sets
X_train = X(1:numTrain, :);
Y_train = Y(1:numTrain, :);

X_test = X(numTrain+1:end, :);
Y_test = Y(numTrain+1:end, :);

% Create iddata objects for training and testing sets
data_train = iddata(Y_train, X_train, 1); % Replace '1' with the actual sampling time
data_test = iddata(Y_test, X_test, 1);


% Define the range for model orders and delays
na = 0:9;  % Number of past outputs (typically start small, e.g., 1 to 5)
nb = 0:9;  % Number of past inputs (same range as above)
nk = 0:9;  % Input delay (start with 0 if unknown)

% Generate the possible model structures
NN = struc(9, 0, 200);

% Evaluate model structures using the `arxstruc` function
V = arxstruc(data_train, data_train, NN);

% Select the optimal model structure based on the loss function
order = selstruc(V, 0); % '0' selects the model structure with the minimum loss function
disp('Optimal Model Structure (na, nb, nk):');
disp(order);

% Build the ARX model using the selected structure
model = arx(data_train, order);

% Simulate or predict the output using the test input data
Y_test_sim = sim(model, X_test);

% Create an iddata object for the simulated output for comparison
data_test_sim = iddata(Y_test_sim, X_test, 1);

% Plot the original test output and simulated output
figure;

% Plot original test output
plot(data_test.y, 'b'); % Original test output in blue
hold on;

% Plot simulated model output
plot(data_test_sim.y, 'r');
legend('Original Test Output', 'Simulated Output');
xlabel('Time');
ylabel('Output Signal');
title('Original vs. Simulated Model Output (Testing Data)');

%%
% Convert ARX model to transfer function
transfer_function = tf(model);

% Display the transfer function
disp('Transfer Function of the ARX Model:');
transfer_function

% Bode plot of the transfer function
figure;
bode(transfer_function);
title('Bode Plot of the ARX Model');

% Define the time vector for the simulation
t = 0:1:500;  % Time from 0 to 10 seconds with a step of 0.1 seconds

% Define the step input (scaled by 5)
u = 5*ones(size(t));  % A step input of magnitude 5

% Simulate the response of the ARX model to the scaled step input
[y, t_out] = lsim(model, u, t);  % model is your ARX model

pzplot(transfer_function)
is_stable = isstable(model);
disp(['Is the system stable? ', num2str(is_stable)]);

% Plot the response
figure;
plot(t_out, y, 'r', 'LineWidth', 1.5);
title('Response of ARX Model to a 5*Step Input');
xlabel('Time (s)');
ylabel('Output');
grid on;
