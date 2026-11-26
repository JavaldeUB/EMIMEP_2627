filename = 'organized_data_with_baseline_paper.csv'; % Specify the name of the CSV file
data = readtable(filename); % Open and read the CSV file into a table

%% Real part modeling.
class_col = data{:,202};

% Calibration concentrations (0 excluded)
trainConcentrations = [20, 40, 50, 60, 80, 96];

% External validation concentrations
testConcentrations = [10, 30, 70];

% Select calibration and external validation samples
keep_mask = ismember(class_col, trainConcentrations);
keep_mask_infer = ismember(class_col, testConcentrations);

data_1 = data(keep_mask,:);
data_infer_1 = data(keep_mask_infer,:);

% Convert to numerical arrays
data_mod = abs(table2array(data_1));
data_test = abs(table2array(data_infer_1));

% Calibration dataset
XTrain = -data_mod(:,1:201);
yTrain = data_mod(:,202);

% External validation dataset
XTest = -data_test(:,1:201);
yTest = data_test(:,202);

Xp = [XTrain; XTest];
yp = [yTrain; yTest];
uniqueNumbers = sort(unique(yp));
counts = nonzeros(histcounts(yp.', 0:0.1:uniqueNumbers(end)));
minCount = min(counts);

freq = linspace(1.6,3,201);             %Frequency Axis
freqAxisTrain = freq(1:201);

%% Plotting the Raw data
hraw = plot_S21Mag(Xp, yp, freqAxisTrain);

% Asymetric Least Squares for baseline removal and plotting
z = zeros(size(Xp));
for i = 1:size(Xp,1)
    z(i,:)=asymmetric_least_squares(-Xp(i,:),1e6,0.001,10);
end
X_ALS = -(-Xp - z);

hals = plot_S21Mag(X_ALS,yp,freqAxisTrain);

%% Preprocessing: 
%1) Cross correlating with means
%Separating dataset
[NTrain, ~] = size(XTrain);
X_Train = X_ALS(1:NTrain,:);
y_Train = yp(1:NTrain);
X_Test = X_ALS(NTrain+1:end,:);
y_Test = yp(NTrain+1:end);

X_mean = mean(X_Train);

X_Train = X_Train - X_mean;
havgTrain = plot_S21Mag(X_Train,y_Train,freqAxisTrain);
X_Test = X_Test - X_mean;
havgTest = plot_S21Mag(X_Test,y_Test,freqAxisTrain);
%% Partitioning the Training dataset for internal validation
groups = y_Train(:);
uniqueGroups = unique(groups);

disp('Concentrations used in LOGO:')
disp(uniqueGroups)

%% LOGO Cross-Validation
% Define groups using real concentration values
groups = y_Train(:);
uniqueGroups = unique(groups);

% Check number of samples per concentration
[levels,~,groupIdx] = unique(groups);
countsPerGroup = accumarray(groupIdx,1);

disp(table(levels,countsPerGroup,...
    'VariableNames',{'Concentration','NumberOfSamples'}));

% Maximum number of PLS components
minTrainSamples = min(arrayfun(@(g) sum(groups ~= g),uniqueGroups));
maxComponents = min([21,size(X_Train,2),minTrainSamples-1]);

% Initialize variables
rmseLOGO = zeros(maxComponents,1);
rmsePooled = zeros(maxComponents,1);

vipScoresMean = zeros(maxComponents,size(X_Train,2));

%% LOGO loop

for nComp = 1:maxComponents

    nGroups = length(uniqueGroups);

    rmseGroup = zeros(nGroups,1);

    vipScores = zeros(nGroups,size(X_Train,2));

    % Store predictions in their original positions
    Y_pred_tot = nan(size(y_Train));

    for i = 1:nGroups

        % Leave one concentration out
        testIdx = groups == uniqueGroups(i);
        trainIdx = ~testIdx;

        X_train_logo = X_Train(trainIdx,:);
        Y_train_logo = y_Train(trainIdx);

        X_test_logo = X_Train(testIdx,:);
        Y_test_logo = y_Train(testIdx);

        % PLS regression
        [XL,YL,XS,YS,beta,PCTVAR,MSE,PLSstats] = ...
            plsregress(X_train_logo,Y_train_logo,nComp);

        % VIP calculation
        W0 = bsxfun(@rdivide,PLSstats.W,...
            sqrt(sum(PLSstats.W.^2,1)));

        sumSq = sum(XS.^2,1).*sum(YL.^2,1);

        vip = sqrt(size(XL,1)*...
            sum(bsxfun(@times,sumSq,W0.^2),2)./sum(sumSq));

        vipScores(i,:) = vip(:)';

        % Prediction
        Y_pred_logo = [ones(size(X_test_logo,1),1),...
            X_test_logo]*beta;

        % Save predictions in original positions
        Y_pred_tot(testIdx) = Y_pred_logo;

        % RMSE for each concentration
        rmseGroup(i) = sqrt(mean(...
            (Y_test_logo-Y_pred_logo).^2));

    end

    % Average RMSE across concentrations
    rmseLOGO(nComp) = mean(rmseGroup);

    % Pooled RMSE across all predictions
    rmsePooled(nComp) = sqrt(mean(...
        (y_Train-Y_pred_tot).^2));

    % Average VIP
    vipScoresMean(nComp,:) = mean(vipScores,1);

end


%% Plotting Cross Validation results
figure;
plot(1:maxComponents,rmseLOGO,'-o');
xlabel('Number of Components');
ylabel('LOGO RMSE');
title('LOGO Cross-Validation');
grid on;

%% PERFORMANCE DURING CROSS-VALIDATION

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%% PERFORMANCE DURING LOGO CROSS VALIDATION %%%%%%%%%%%%%%%%%%%');

% Select optimal number of PLS components
[~, optimalComponents] = min(rmseLOGO);

disp(['Optimal Number of Components: ', num2str(optimalComponents)]);

%% Reconstruct LOGO predictions using optimal components

groups = y_Train(:);
uniqueGroups = unique(groups);

% Initialize predictions
Y_pred_CV = zeros(size(y_Train));

for i = 1:length(uniqueGroups)

    % Leave-one-group-out partition
    testIdx = (groups == uniqueGroups(i));
    trainIdx = ~testIdx;

    X_train_logo = X_Train(trainIdx,:);
    Y_train_logo = y_Train(trainIdx);

    X_test_logo = X_Train(testIdx,:);
    Y_test_logo = y_Train(testIdx);

    % PLS model for each fold
    [~,~,~,~,betaCV] = plsregress( ...
        X_train_logo,Y_train_logo,optimalComponents);

    % Prediction of the excluded group
    Y_pred_logo = [ones(size(X_test_logo,1),1), ...
        X_test_logo]*betaCV;

    % Store predictions in original sample positions
    Y_pred_CV(testIdx) = Y_pred_logo;

end

%% Calculate Cross-Validation Performance

RMSECV = sqrt(mean((y_Train-Y_pred_CV).^2));

RSS = sum((y_Train-Y_pred_CV).^2);
TSS = sum((y_Train-mean(y_Train)).^2);

r2 = 1-(RSS/TSS);

disp(['Cross-Validation RMSECV: ', num2str(RMSECV)]);
disp(['Cross-Validation R^2: ', num2str(r2)]);


%% Calculate Bias and Limits of Agreement

bias = mean(Y_pred_CV-y_Train);

LoA = 1.96*std(Y_pred_CV-y_Train);

upperLoA = bias+LoA;
lowerLoA = bias-LoA;

disp(['Cross-Validation Bias: ', num2str(bias)]);
disp(['Cross-Validation Upper LoA: ', num2str(upperLoA)]);
disp(['Cross-Validation Lower LoA: ', num2str(lowerLoA)]);


%% Predicted vs Actual - Cross Validation

minVal = min([y_Train;Y_pred_CV]);
maxVal = max([y_Train;Y_pred_CV]);

figure;

scatter(y_Train,Y_pred_CV,'b','filled');
hold on;

% Perfect agreement line
plot([minVal,maxVal],[minVal,maxVal], ...
    'k','LineWidth',1.5);

xlabel('Actual Concentration (CV)');
ylabel('Predicted Concentration');

title('Predicted vs Actual - LOGO Cross Validation');

legend('Predicted vs Actual', ...
    'Perfect Agreement','Location','Best');

grid on;
axis([0 100 0 100]);

hold off;


%% Bland-Altman Plot - Cross Validation

meanCV = (y_Train+Y_pred_CV)/2;
diffCV = Y_pred_CV-y_Train;

bias = mean(diffCV);
LoA = 1.96*std(diffCV);

upperLoA = bias+LoA;
lowerLoA = bias-LoA;

figure;

scatter(meanCV,diffCV,'b','filled');
hold on;

minBA = min(meanCV);
maxBA = max(meanCV);

% Bias line
plot([minBA,maxBA],[bias,bias], ...
    'r','LineWidth',1.5);

% Upper and Lower Limits of Agreement
plot([minBA,maxBA],[upperLoA,upperLoA], ...
    'k-.','LineWidth',1.5);

plot([minBA,maxBA],[lowerLoA,lowerLoA], ...
    'k-.','LineWidth',1.5);

xlabel('Mean of Actual and Predicted Concentration');
ylabel('Predicted - Actual Concentration');

title('Bland-Altman - LOGO Cross Validation');

legend('Differences','Bias','Upper LoA','Lower LoA', ...
    'Location','Best');

grid on;
hold off;


%% PERFORMANCE DURING EXTERNAL VALIDATION

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%% PERFORMANCE DURING EXTERNAL VALIDATION %%%%%%%%%%%%%%%%%%%%');

%% Final PLS model

% Train final model using all calibration samples
[XL,YL,XS,YS,betaFinal,PCTVAR,MSE,PLSstatsF] = ...
    plsregress(X_Train,y_Train,optimalComponents);


%% External prediction

Y_pred_test = [ones(size(X_Test,1),1),X_Test]*betaFinal;


%% Calculate External Validation Performance

RMSEP = sqrt(mean((y_Test-Y_pred_test).^2));

RSSinterp = sum((y_Test-Y_pred_test).^2);
TSSinterp = sum((y_Test-mean(y_Test)).^2);

r2interp = 1-(RSSinterp/TSSinterp);

disp(['External Validation RMSEP: ',num2str(RMSEP)]);
disp(['External Validation R^2: ',num2str(r2interp)]);


%% Calculate Bias and Limits of Agreement

bias = mean(Y_pred_test-y_Test);

LoA = 1.96*std(Y_pred_test-y_Test);

upperLoA = bias+LoA;
lowerLoA = bias-LoA;

disp(['External Validation Bias: ',num2str(bias)]);
disp(['External Validation Upper LoA: ',num2str(upperLoA)]);
disp(['External Validation Lower LoA: ',num2str(lowerLoA)]);

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');


%% Predicted vs Actual - External Validation

minVal = min([y_Test;Y_pred_test]);
maxVal = max([y_Test;Y_pred_test]);

figure;

scatter(y_Test,Y_pred_test,'b','filled');
hold on;

% Perfect agreement line
plot([minVal,maxVal],[minVal,maxVal], ...
    'k','LineWidth',1.5);

xlabel('Actual Concentration (Prediction)');
ylabel('Predicted Concentration');

title('Predicted vs Actual - External Validation');

legend('Predicted vs Actual', ...
    'Perfect Agreement','Location','Best');

grid on;
axis([0 100 0 100]);

hold off;


%% Bland-Altman Plot - External Validation

meanTest = (y_Test+Y_pred_test)/2;
diffTest = Y_pred_test-y_Test;

bias = mean(diffTest);
LoA = 1.96*std(diffTest);

upperLoA = bias+LoA;
lowerLoA = bias-LoA;

figure;

scatter(meanTest,diffTest,'b','filled');
hold on;

minBA = min(meanTest);
maxBA = max(meanTest);

% Bias line
plot([minBA,maxBA],[bias,bias], ...
    'r','LineWidth',1.5);

% Upper and Lower Limits of Agreement
plot([minBA,maxBA],[upperLoA,upperLoA], ...
    'k-.','LineWidth',1.5);

plot([minBA,maxBA],[lowerLoA,lowerLoA], ...
    'k-.','LineWidth',1.5);

xlabel('Mean of Actual and Predicted Concentration');
ylabel('Predicted - Actual Concentration');

title('Bland-Altman - External Validation');

legend('Differences','Bias','Upper LoA','Lower LoA', ...
    'Location','Best');

grid on;
hold off;

%% LoD calculation
b = betaFinal(2:end);
b0 = betaFinal(1);
inv_norm = (b' * b);
for i = 1:NTrain
    rHat(i,:) = ((y_Train(i) - b0) / inv_norm) * b';
end
r = X_Train;   
e = rHat - r;
s_r = sqrt(sum(e.^2,1) / (size(X_Train,1) - 2));
LODp = 3 * abs(s_r*b/norm(b));
disp(['Limit of Detection (LoD): ', num2str(LODp), '%'])

betaEprP = betaFinal;
meanP = X_mean;

