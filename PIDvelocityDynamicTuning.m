%% Dynamic Velocity PID with tuning options 
    % LAST UPDATE: May 21, 2024 - PSC 
    % *** Tuning options added for comparison 
    % * Training - get velocity tuning for each chunk of the trial
    % * Dynamics - 'stiched' velocity tuning as a function of the position 
    %               of the vm in the trial   
    % * Testing  - calculate loglikelihoods, estimate acceleration from
    %              velocity decoding 
    % * Current Ver - 10 x 100ms chunks, 10 runs
    % * 'stiched' Tuning options - 1) dynamic, 2) transient, 3) full trial

%% (1) Calculate the midspds for each chunk for all the trials, and get uniqueVms 
%nunits  = dv.tune.nunits;
nchunks = size(dv.tune.trspd,3);
rate    = dv.tune.rate;
nrun    = 10;
chunkInt = 0.10;
condVar;
 % tuning curve options 
    % 1 = dynamic velocity tuning; 
    % 2 = dynamic transient-velocity; 
    % 3 = full trial velocity tuning;
%tuning = 1; 

% store the trial Ve and V0 index and acc index
    condIdxspd(:,1) = rate.condIdx(:,1); % Ve
    condIdxspd(:,2) = rate.condIdx(:,2); % V0
    condIdxspd(:,3) = rate.condIdx(:,3); % Acc

% start & end velocities for each chunk, initial spd of the trial 
    startspds = zeros(size(dv.tune.trspd,[1,3])); % chunk start velocity  
    endspds   = zeros(size(dv.tune.trspd,[1,3])); % chunk end velocity 
    midspds   = zeros(size(dv.tune.trspd,[1,3])); % chunk mid velocity 

for i = 1:size(dv.tune.trspd,1)
    for c = 1:nchunks
        endspds(i,c)   = round(v0(condIdxspd(i,2)) + uniAcceleration(condIdxspd(i,3)).*(0.2.*c),3,'significant');
        startspds(i,c) = round(v0(condIdxspd(i,2)) + uniAcceleration(condIdxspd(i,3)).*(0.2.*(c-1)),3,'significant');
        midspds(i,c)   = round(startspds(i,c) + (uniAcceleration(condIdxspd(i,3)).*0.1),3,'significant');
    end
end

% get the ubique Vm conditions 
Vm = unique(midspds);
nVm = length(Vm);

% midVels condIdx for ALL trials 
for i = 1:length(dv.tune.trspd)
    for c = 1:nchunks
        theidx = find(Vm == midspds(i,c)); % theidx = the index in the unique Vm vector 
        condIdxmidspd(i,c) = theidx;        
    end 
end 

% save vm index dv structure 
dv.tune.condIdxmidspd = condIdxmidspd;
dv.tune.uniVm = Vm;

%% (2) Get the needed index vectors/matrices based on trial selection 
ratevel = dv.tune.trspd;

idxNR = cell2mat(idxNRA); % - all NR trials available 
idxSELECT;                % - the 10% testing trials for 10 runs 
idxRUN;                   % - the 90% training trials for 10 runs 

condIdxmidspd_NR = condIdxmidspd(idxNR,:);
condIdxVm_Train  = {}; % vm condIdx for training trials 
condIdxVm_Test   = {}; % vm condIdx for testing trials 

for run = 1:nrun
    for i = 1:length(idxRUN{run})
        condIdxVm_Train{run}(i,:,:) = condIdxmidspd(idxRUN{run}(i),:);
    end

    for i = 1:length(idxSELECT{run})
        condIdxVm_Test{run}(i,:,:) = condIdxmidspd(idxSELECT{run}(i),:);
    end
end

%% (3) Organize the available trials

% set the SRs cell for training and testing trials 
trNRtrain = {};
trNRtest = {};
trNRFullTune = {};

for run = 1:nrun
    for i = 1:length(idxRUN{1})
        trNRtrain{run}(i,:,:) = ratevel(idxRUN{run}(i),:,:);
        trNRFullTune{run}(i,:) = rate.raw(idxRUN{run}(i),:);
    end

    for i = 1:length(idxSELECT{1})
        trNRtest{run}(i,:,:) = ratevel(idxSELECT{run}(i),:,:);
        
    end
    
end

% might be helpful...who knows 
ntr_NR = zeros(nVm,1);
for i = 1:nVm
    for row = 1:length(condIdxmidspd_NR)
        if any(condIdxmidspd_NR(row,:)==i)
            ntr_NR(i) = ntr_NR(i) + 1;
        end
    end
end


%% (4) Add to the current dv structure 

dv.tune.condIdxVm_Train = condIdxVm_Train;
dv.tune.condIdxVm_Test = condIdxVm_Test;
dv.tune.trNRtrain = trNRtrain;
dv.tune.trNRtest = trNRtest;
dv.tune.trNRFullTune  = trNRFullTune;

%% (5) Run dynamic velocity decoder

clear logL;

for run = 1:nrun
   
    vmidxtrain = squeeze(condIdxVm_Train{run});
    testSR = dv.tune.trNRtest{run};

    % get the stiched tuning for this run 
    [f]=generateETuningVEL(dv,nVm,vmidxtrain,run,nchunks,tuning);

    % PID set up here
    for i = 1:length(testSR)
        for c = 1:nchunks

            for u = 1:nunits
                r(u) = testSR(i,u,c);
            end

            for vmid = 1:nVm
                % calculate the loglikelihood 
                logL(i,vmid,c,run) = logL_vel(f,r,vmid,dv,c);               
            end
        end
    end 

end


%% (6) Calculate PID decoding results
clear resultVm; clear decodeVm; clear logLval; clear estvmCond; clear actualAcc;

% Decoding results of velocity  
for run = 1:nrun
    for i = 1:size(logL,1)
        for c = 1:nchunks
            [logLval, estvmCond] = max(logL(i,:,c,run));
            resultVm(i,c,run) = estvmCond;
            decodeVm(i,c,run) = Vm(resultVm(i,c,run));
        end 
    end 
end 

% Real acceleration condition 
actualAcc = uniAcceleration((condIdxspd(idxSELECT{1},3)));

%% (7) Get the linear fits  
% fit5/fit4/fit       - fits before averaging across runs
% fit5Vm/fit4Vm/fitVm - fits averaged across runs 
clear x; clear y;clear fit; clear fit5; clear fit4; clear fitVm; clear fit5Vm; 
clear fit4Vm;

for run =1:nrun
    % (1) fit with ALL chuncks
    for i = 1:size(decodeVm,1)
        y(i,1:nchunks) = decodeVm(i,1:nchunks,run);
        x = 0.05:chunkInt:0.95;
        fit5(i,:,run) = polyfit(x,y(i,:),1);
    end

    clear x; clear y;
    % (2) fit with later 4 chuncks
    for i = 1:size(decodeVm,1)
        y(i,1:nchunks-1) = decodeVm(i,2:nchunks,run);
        x = chunkInt+0.05:chunkInt:0.95;
        fit4(i,:,run) = polyfit(x,y(i,:),1);

    end

    clear x; clear y;
    % (3) fit for individual chuncks
    for i = 1:size(decodeVm,1)
        for k = 1:(nchunks-1)
            y(i,1:2) = decodeVm(i,k:k+1,run);
            x = [(k-1).*chunkInt+0.05, k.*chunkInt+0.05];
            fit(i,:,k,run) = polyfit(x,y(i,:),1);
        end
    end
end

% average across the runs 
fit5Vm = nanmean(fit5,3);
fit4Vm = nanmean(fit4,3);
fitVm  = nanmean(fit,4);

%% (11)*** SAVE decoding results save back to the dv structures***
dv.decode.decodeVel90 = decodeVm;
dv.decode.actualVAcc90 = actualAcc;
dv.decode.fitVm90 = fitVm;
dv.decode.fitNTVm90 = fit4Vm;
dv.decode.fitALLVm90 = fit5Vm;

%% (12) Statistics & Save stats 
% (1) R-squared - against the unity line 
clear accActual; clear accDecoded; clear RSS; clear TSS; clear Rsquared;
clear id; clear btsR2CI; clear btsRsquared; 

% (12.1) R-squared - against the unity line 
% setting up 
nit = 10000;
ndraw = length(dv.decode.actualVAcc90); % draw with replacement 
id = ceil(rand(ndraw,nit)*ndraw);
CIrange = 95;

% R2 with bootstrapping 
for it = 1:nit
    
    accActual = dv.decode.actualVAcc90(id(:,it),:);
    
    for i = 1:11
        if i <=9
            accDecoded = dv.decode.fitVm90(id(:,it),1,i);
        elseif i== 10
            accDecoded = dv.decode.fitALLVm90(id(:,it),1);
        elseif i == 11
            accDecoded = dv.decode.fitNTVm90(id(:,it),1);
        end
        
        % Sum of squared residuals
        RSS = sum((accDecoded - accActual).^2);
        % Total sum of squares
        TSS = sum(((accDecoded - mean(accDecoded)).^2));
        % R squared
        btsRsquared(i,it) = 1 - RSS/TSS;
    end
end

mubtsRsquared = nanmean(btsRsquared,2);

% compute CIs 
for i = 1:size(btsRsquared,1)
    btsR2CI(i,:) = prctile(btsRsquared(i,:),[50-CIrange/2,50+CIrange/2]);
end

%% (12.2) Add the basic stats back to dv structure 
dv.stats.dynVmbtsR2U = btsRsquared;
dv.stats.dynVmbtsmuR2U = mubtsRsquared;
dv.stats.dynVmbtsR2CIU = btsR2CI;

%% (13) Best fit line from the regression models 
clear accActual; clear accDecoded; clear mdl;clear id; clear ndraw;clear vm;
clear am; clear muR2; clear R2CI; clear VM;
%nunits = dv.tune.nunits;
nit = 1000;
ndraw = length(dv.decode.actualVAcc90); % draw with replacement 
id = ceil(rand(ndraw,nit)*ndraw);
CIrange = 95;

% fitlm with bootstrapping 
parfor it = 1:nit
    
    accActual = dv.decode.actualVAcc90(id(:,it),:);
   
    for i = 1:11
        if i <=9
            accDecoded = dv.decode.fitVm90(id(:,it),1,i);
        elseif i==10
            
            accDecoded = dv.decode.fitALLVm90(id(:,it),1);
        elseif i ==11
            accDecoded = dv.decode.fitNTVm90(id(:,it),1);
        end
        mdl = fitlm(accActual,accDecoded);
        vm{i,it} = struct('coef',{mdl.Coefficients},'R2',{mdl.Rsquared});
        
    end
end

for i = 1:11
    for it = 1:nit 
       VM(i,it) = vm{i,it}.R2.Adjusted;
    end
end 

muR2 = nanmean(VM,2)'
% compute CIs 
for i = 1:size(VM,1)
    R2CI(i,:) = prctile(VM(i,:),[50-CIrange/2,50+CIrange/2]);
end


% (13.b) - save the basic stats back to the dv structure
dv.stats.dynVmbtsmuR2 = muR2;
dv.stats.dynVmbtsR2 = VM;
dv.stats.dynVmbtsR2CI = R2CI;

%% (8) !!!!!!! PLOTs !!!!!!!
% (a) plot the R2 distributions & CIs  
figure(6666); 
ax1 = subplot(221); 
histogram(btsRsquared(5,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(5,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(5,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('1sec');
ax2 = subplot(222); 
histogram(btsRsquared(6,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(6,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(6,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('200-1000ms');
ax3 = subplot(2,4,5);
histogram(btsRsquared(1,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(1,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(1,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('0-400ms');
ax4 = subplot(2,4,6);
histogram(btsRsquared(2,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(2,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(2,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('200-600ms');
ax5 = subplot(2,4,7);
histogram(btsRsquared(3,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(3,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(3,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('400-800ms');
ax6 = subplot(2,4,8);
histogram(btsRsquared(4,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(4,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(4,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('600-1000ms');
linkaxes([ax1,ax2,ax3,ax4,ax5,ax6],'y');
sgtitle('Bootstrapped 95% CIs fo R2')
%% (b) velocity decoding - best line of fit 
figure(7777);
ax1 = subplot(321); 
scatter(actualAcc,fit5Vm(:,1)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('full stimulus presentation - 1sec, R2 = ',num2str(muR2(10))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax2 = subplot(322); 
scatter(actualAcc,fit4Vm(:,1)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('stimulus presentation 200 - 1000ms, R2 = ',num2str(muR2(11))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax1,ax2],'xy');

ax3 = subplot(3,5,6); 
scatter(actualAcc,fitVm(:,1,1)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('0-200ms, R2 = ',num2str(muR2(1))))

ax4 = subplot(3,5,7); 
scatter(actualAcc,fitVm(:,1,2)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim); axis('square');
title(strcat('100-300ms, R2 = ',num2str(muR2(2))))

ax5 = subplot(3,5,8); 
scatter(actualAcc,fitVm(:,1,3)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('200-400ms, R2 = ',num2str(muR2(3))))

ax6 = subplot(3,5,9); 
scatter(actualAcc,fitVm(:,1,4)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('300-500ms, R2 = ',num2str(muR2(4))))

ax7 = subplot(3,5,10); 
scatter(actualAcc,fitVm(:,1,5)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('400-600ms, R2 = ',num2str(muR2(5))))

ax8 = subplot(3,5,11); 
scatter(actualAcc,fitVm(:,1,6)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('500-700ms, R2 = ',num2str(muR2(6))))

ax9 = subplot(3,5,12); 
scatter(actualAcc,fitVm(:,1,7)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('600-800ms, R2 = ',num2str(muR2(7))))

ax10 = subplot(3,5,13); 
scatter(actualAcc,fitVm(:,1,8)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('700-900ms, R2 = ',num2str(muR2(8))))

ax11 = subplot(3,5,14); 
scatter(actualAcc,fitVm(:,1,9)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('800-1000ms, R2 = ',num2str(muR2(9))))
linkaxes([ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11],'xy');
sgtitle("MT - Dynamic Velocity Decoder 90/10")


%% (9) Function - generate the weights = velocity tuning  

% f = ensemble velocity tuning
function [f] = generateETuningVEL(dv,nVm,vmidxtrain,run,nchunks,tuning)

% set up the indices
trainSR = dv.tune.trNRtrain{run};
trainFTSR = dv.tune.trNRFullTune{run};


% set up tuning functions
f = zeros(nVm,dv.tune.nunits,nchunks);
%f = zeros(nVm,51,nchunks);

% get the 'stiched' tuning for each vm condition
for vm=1:nVm
    for c = 1:nchunks
        idx = find(vmidxtrain(:,c)==vm);
        if ~isempty(idx)
            if tuning == 1
                f(vm,:,c) = squeeze(nanmean(trainSR(idx,:,c),1));
            elseif tuning == 2
                f(vm,:,c) = squeeze(nanmean(trainSR(idx,:,1),1));
            elseif tuning ==3
                f(vm,:,c) = squeeze(nanmean(trainFTSR(idx,:),1));
            end
        end
    end
end

end

%% (10) Function - calculate the log-likelihood

function logL = logL_vel(f,r,vmCond,dv,c)

for u = 1:dv.tune.nunits
    fu(u) = f(vmCond,u,c);
    logfru(u) = log(fu(u)).*r(u);
    logru(u) = logfactorial(r(u));
end

logL = sum(logfru) - sum(fu) - sum(logru);

end

