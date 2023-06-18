%% Inititalize data
clc
clear
load sympy.mat
x = sympy_data(:,1);
y = sympy_data(:,2);
z = sympy_data(:,3);

%% Grid data
[xq,yq] = meshgrid(0:.01:20, 0:.01:20);
vq = griddata(x,y,z,xq,yq,'natural');
figure
mesh(xq,yq,vq)
colormap(winter)
colorbar
hold on
plot3(x,y,z,'r*')
grid on
grid minor
hold off
xlabel('Degree of numerator')
ylabel('Degree of denominator')
title('Abortion rate for 7 sec', 'FontSize', 14)
%% Heatmap
% Convert the data to a table
T = table(x, y, z);

figure
% Create a heatmap using the heatmap function
heatmap(T, 'x', 'y', 'ColorVariable', 'z', 'Colormap', colormap(winter))

% Set axis labels and title
xlabel('Degree of numerator')
ylabel('Degree of denominator')
title('0.1 sec Sample size factor')