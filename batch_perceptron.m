% Batch Perceptron 
% Author: Amir Azarbakht  <azarbaam@eecs.oregonstate.edu>
% Date: 2014-04-15

clear all;
close all;

% import the data
data = load('twogaussian.csv');
[dim1, dim2] = size(data);
% extract first column as class labels
Y = data(:,1);
% add dummy feature for w0, and extract the rest of the column as X
X = [ones(dim1,1), data(:,2:end);];

clear dim1 dim2;
[dim1, dim2] = size(X);

w = zeros(1,dim2);
delta = zeros(1,dim2);
epsil = 0.0000001;
g = ones(dim1,1);

MAX_epoch = 100;
epoch_count = 0;
error_count = 0;
epoch_error = zeros(MAX_epoch,2);

while true 
    delta = zeros(1,dim2);
    
    epoch_count = epoch_count + 1;
    error_count = 0;
    
    for i = 1:dim1,
        
        g(i,1) = w * X(i,:)';
        if Y(i,1)*g(i,1) <= 0
           delta = delta - (Y(i,1)*X(i,:));
           error_count = error_count + 1;
        end
        epoch_error(epoch_count, 2) = error_count;
    end
    
    epoch_error(epoch_count, 1) = epoch_count;
    epoch_error(epoch_count, 2) = error_count;
    
    delta = delta / dim1;
    w = w - delta;     
    if max(abs(delta)) < epsil ; break; end 
end

w
epoch_count
epoch_error
csvwrite('fig2_batch_perceptron.csv',w);

% X(X<0)=0; % Replaces all negative entries of X with 0
temp = g;
temp(sign(temp)==1)=1;
temp(sign(temp)~=1)=-1;
g = temp;

% scatter(getcolumn(data(:,2:end),1),getcolumn(data(:,2:end),2));figure(gcf)

j = 0;
k = 0;
for i = 1:size(data,1),
    if data(i,1) > 0
        j = j + 1;
        posData(j,:) = data(i,2:end);
    else
        k = k + 1;
        negData(k,:) = data(i,2:end);
    end
end

% scatter plot of the data, and the learned linear classifier
figure1 = figure('Color',[1 1 1]);
figure(1);
scatter(getcolumn(negData,1),getcolumn(negData,2),'r', 'o');
figure(gcf)
hold on; 
scatter(getcolumn(posData,1),getcolumn(posData,2),'b', 'x');
figure(gcf)
legend('Positive class', 'Negative class');
ezplot([num2str(w(1)) ' + ' num2str(w(2)) '* x ' num2str(w(3)) '* y = 0' ,]);
xlabel('x_1');
ylabel('x_2');
title('Batch Perceptron');


saveas(1, 'Batch_Perceptron', 'png');
saveas(1, 'Batch_Perceptron', 'eps');
saveas(1, 'Batch_Perceptron', 'fig');

% bar chart of classification error as a funtion of the number of epoches
figure2 = figure('Color',[1 1 1]);
figure(2);
plot(epoch_error(1:end,end), 'r');figure(gcf);
title('Batch Perceptron: Classification Error as a function of the number of training epoches');
xlabel('Number of training epoches');
ylabel('Classification Error');
saveas(2, 'Batch_Perceptron_Classification_Error_vs_numberOfTrainingEpoches', 'png');
saveas(2, 'Batch_Perceptron_Classification_Error_vs_numberOfTrainingEpoches', 'fig');
saveas(2, 'Batch_Perceptron_Classification_Error_vs_numberOfTrainingEpoches', 'epsc2');
