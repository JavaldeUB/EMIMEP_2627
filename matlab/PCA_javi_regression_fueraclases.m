filename = 'organized_data_with_baseline_paper.csv'; % Specify the name of the CSV file
data = readtable(filename); % Open and read the CSV file into a table

% Define the 3 classes to remove from column 202 (adjust values as needed)
classes_to_remove = [80,50,30];  % Example: remove classes 1, 2, and 3
class_col = data{:, 202};           % Extract column 202 as vector

%% Create logical mask to keep rows not matching the classes to remove
keep_mask = ~ismember(class_col, classes_to_remove);
keep_mask_infer=ismember(class_col, classes_to_remove);

%Apply filter to the table
data_1 = data(keep_mask, :); % Remove rows belonging to the 3 classes
data_infer_1=data(keep_mask_infer, :); % Save rows belonging to the 3 classes
data_mod = abs(data_1);
data_test = abs(data_infer_1);
data_mod = table2array(data_mod);
data_test = table2array(data_test);
data_norm=[data_mod;data_test];

%% Preprocessing: 
% Add your own code to the block.
data_mod(:,1:201) = (data_mod(:,1:201));
data_test(:,1:201) = (data_test(:,1:201));

%% Plot Dataset
% First the modelling set
freq = linspace(1.6,3,201);             %Frequency Axis
freqAxis = freq(1:201);
uniqueNumbers = unique(data_mod(:,202));
colors = jet(length(uniqueNumbers));
legend_handles = gobjects(length(uniqueNumbers), 1);
figure;
hold on;
for i = 1:length(uniqueNumbers)
    label = uniqueNumbers(i);   
    rows = data_mod(data_mod(:,202) == label, 1:201);
    for j = 1:size(rows, 1)   
        h = plot(freqAxis,(-(rows(j, :))), 'Color', colors(i, :));        
        if j == 1
           legend_handles(i) = h;
        end
    end
end
hold off;
xlabel("Freq (GHz)");
ylabel("20log_{10}|S_{21}|");
title("Modelling samples");
legend(legend_handles, arrayfun(@num2str, uniqueNumbers,...
    'UniformOutput', false), 'Location', 'best');

% second the test set
freq = linspace(1.6,3,201);             %Frequency Axis
freqAxis = freq(1:201);
uniqueNumbers = unique(data_test(:,202));
colors = jet(length(uniqueNumbers));
legend_handles = gobjects(length(uniqueNumbers), 1);
figure;
hold on;
for i = 1:length(uniqueNumbers)
    label = uniqueNumbers(i);   
    rows = data_test(data_test(:,202) == label, 1:201);
    for j = 1:size(rows, 1)   
        h = plot(freqAxis,(-(rows(j, :))), 'Color', colors(i, :));        
        if j == 1
           legend_handles(i) = h;
        end
    end
end
hold off;
xlabel("Freq (GHz)");
ylabel("20log_{10}|S_{21}|");
title("Test samples");
legend(legend_handles, arrayfun(@num2str, uniqueNumbers,...
    'UniformOutput', false), 'Location', 'best');


%% PCA
num_coef=8;
[coeff, score, latent, tsquared, explained, mu_train] =...
    pca(data_mod(:,1:201), 'NumComponents', num_coef);

% Plot explained variance
figure;
bar(explained, 'FaceAlpha', 0.7);
title('Explained Variance Distribution by Principal Components');
xlabel('Principal Component');
ylabel('Explained Variance (%)');
xlim([0 num_coef+1]);
grid on;
figure;
plot(cumsum(explained), '-o', 'LineWidth', 1.5, 'DisplayName', 'Cumulative Explained Variance');
xlabel('Principal Component');
ylabel('Explained Variance (%)');
title('Explained Variance by Principal Components');
legend;
grid on;

% 2D Scatter plot: PC2 vs. PC1
figure;
yp = data_mod(:,202);
scatter(cat(1,score(1:20, 1),score(22:65, 1),score(67:end, 1)), cat(1,score(1:20, 2),score(22:65, 2),score(67:end, 2)),...
    50, cat(1,yp(1:20, 1),yp(22:65, 1),yp(67:end, 1)), 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PC2 vs. PC1');
colormap('jet');
colorbar;
grid on;


% 2D Scatter plot: PC3 vs. PC1
figure;
scatter(score(:, 1), score(:, 3), 50, yp, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 3');
title('PC3 vs. PC1');
colormap('jet');
colorbar;
grid on;
hold off;

% 3D Scatter plot: PC1, PC2, and PC3
figure;
scatter3(score(:, 1), score(:, 2), score(:, 3), 50, yp, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('3D Plot of PC1, PC2, and PC3');
colormap('jet');
colorbar;
grid on;

%% Data Proyection to PCA space
data = score;
X = data_mod(:, 1:201);
Y = data_mod(:, 202);

X_infer = data_test(:, 1:201);
Y_infer = data_test(:, 202);


%% PLS regression: Cosas a cambiar por ellos
% Hacer una optimización del número de componentes.
ncomp=8;
[Xl,Yl,Xs,Ys,beta,PCTVAR,mse,stats] = plsregress(X,Y,ncomp);

% Prediction
A=[ones(size(X_infer,1),1) X_infer];
Yfit = [ones(size(X_infer,1),1) X_infer] * beta;



% Calcular el RMSE (root mean squared error)
rmse = rms(sqrt(mean((Yfit - Y_infer).^2)));
disp(['RMSE PLS: ' num2str(rmse)]);

%% ==== Cálculo del MAPE (error porcentual absoluto medio) ====
% Crear máscara para evitar división por cero
mask = abs(Y_infer) > 1e-10;

mape_per_feat = nan(1, size(Y_infer,2));
for feat = 1:size(Y_infer,2)
    y_true_col = Y_infer(:, feat);
    y_pred_col = Yfit(:, feat);
    mask_col = mask(:, feat);
    if nnz(mask_col) > 0
        mape_per_feat(feat) = mean(abs((y_true_col(mask_col) - y_pred_col(mask_col))./ y_true_col(mask_col)))*100;
    end
end

avg_mape = mean(mape_per_feat, 'omitnan');

disp(['Mean RMSE (%): ', num2str(avg_mape)]);

% ==== R² GLOBAL ====
Yfit_flat = Yfit(:);
Y_infer_flat = Y_infer(:);
SS_res_global = sum((Yfit_flat - Y_infer_flat).^2);
SS_tot_global = sum((Y_infer_flat - mean(Y_infer_flat)).^2);
R2_global = 1 - (SS_res_global / SS_tot_global);
disp(['R² Global: ', num2str(R2_global)]);


fid = fopen('beta_carray.txt', 'w');
fprintf(fid, 'float beta[%d][%d] = {\n', size(beta,1), size(beta,2));
for i = 1:size(beta,1)
    fprintf(fid, '  {');
    fprintf(fid, '%.8f, ', beta(i,1:end-1));
    fprintf(fid, '%.8f}', beta(i,end));
    if i < size(beta,1)
        fprintf(fid, ',\n');
    else
        fprintf(fid, '\n');
    end
end
fprintf(fid, '};\n');
fclose(fid);

fid = fopen('mu_train_carray.txt', 'w');
fprintf(fid, 'float mu_train[%d] = {', length(mu_train));
fprintf(fid, '%.8f, ', mu_train(1:end-1));
fprintf(fid, '%.8f};\n', mu_train(end));
fclose(fid);

fid = fopen('coeff_carray.txt', 'w');
fprintf(fid, 'float coeff[%d][%d] = {\n', size(coeff,1), size(coeff,2));
for i = 1:size(coeff,1)
    fprintf(fid, '  {');
    fprintf(fid, '%.8f, ', coeff(i,1:end-1));
    fprintf(fid, '%.8f}', coeff(i,end));
    if i < size(coeff,1)
        fprintf(fid, ',\n');
    else
        fprintf(fid, '\n');
    end
end
fprintf(fid, '};\n');
fclose(fid);

average=mean(data_norm(:,1:201));
std_dev=std(data_norm(:,1:201));

fid = fopen('means_carray.txt', 'w');
fprintf(fid, 'float means[%d] = {', length(average));
fprintf(fid, '%.8f, ', average(1:end-1));
fprintf(fid, '%.8f};\n', average(end));
fclose(fid);

fid = fopen('stds_carray.txt', 'w');
fprintf(fid, 'float stds[%d] = {', length(std_dev));
fprintf(fid, '%.8f, ', std_dev(1:end-1));
fprintf(fid, '%.8f};\n', std_dev(end));
fclose(fid);

sample=data_test_a(9, 1:201);
sample_log=20*log10(sample);

fid = fopen('xs_carray.txt', 'w');
fprintf(fid, 'float x_in[%d] = {', length(sample));
fprintf(fid, '%.8f, ', sample(1:end-1));
fprintf(fid, '%.8f};\n', sample(end));
fclose(fid);

