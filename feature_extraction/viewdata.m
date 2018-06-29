clear all; clc;
files = dir('viewdata/*.txt'); 
numFiles = length(files);

for i = 1:numFiles
    fileID = fopen(strcat('viewdata/', files(i).name), 'r');
    figure_title = files(i).name; 
    disp(figure_title);
    formatSpec = '%d %f';
    sizeA = [2 Inf];
    A = fscanf(fileID, formatSpec, sizeA);
    fclose(fileID);
    A = A';
    h = figure();
    plot(A(:, 1), A(:, 2));
    title(figure_title)
end