%% Dynamic Acceleration PID 
% LAST UPDATE: May 20, 2024 - PSC 
% *** Tuning options added
% * Training - get acceleration tuning for each chunk of the trial
% * Dynamics - acceleration tuning as a function of time 
% * Testing  - calculate loglikelihoods, estimate acceleration from
%              acceleration decoding 
% * Current Ver - 10 x 100ms chunks, 10 runs - use this to save to 'dv'
% * Tuning options - 1) dynamic, 2) transient, 3) full trial 

%% (1) Trial selection - take out the reversals & check  
% settings 
% tuning curve options 
    % 1 = dynamic acceleration tuning; 
    % 2 = dynamic transient-acceleration; 
    % 3 = full trial acceleration;
tuning = 1; 
condVar;
nunits  = dv.tune.nunits;
nchunks = size(dv.tune.trspd,3);
chunkInt = 0.10; % chunk interval in sec 
nrun    = 10;
nAcc    = length(uniAcceleration);
rate    = dv.tune.trspd;
   
% get the non-reversal trials 
ai      = dv.tune.rate.condIdx(:,3);
ntr     = hist(ai, nAcc);
[trNRA] = deal(nan([mmax(ntr), nAcc, nunits,nchunks]));
sz      = size(trNRA);
idxNRA  = {};
for i = 1:nAcc
    accid = i;
    for ii = 1:length(dv.tune.rate.condIdx)

        subs(ii) = (dv.tune.rate.condIdx(ii,3)==accid)&&(dv.tune.rate.condIdx(ii,1)>=8)&&(dv.tune.rate.condIdx(ii,2)>=8);

    end

    therows = find(subs);
    idxNRA{i}= therows;
    if ~isempty(therows)
        for nu = 1:nunits
            u = nu;
            trNRA(1:length(therows),i,nu,:) = rate(therows, u,:);
        end
    end
end

% take out the reversals - full trial, no chunk segments 
%nunits = dv.tune.nunits;
rate = dv.tune.rate;
ai = rate.condIdx(:,3);
ntr = hist(ai, 79);
[ctNR trNR] = deal(nan([mmax(ntr), 79, nunits]));
sz = size(ctNR);
for i = 1:79
    accid = i;
    
    for ii = 1:length(rate.condIdx)
        
        subs(ii) = (rate.condIdx(ii,3)==accid)&&(rate.condIdx(ii,1)>=8)&&(rate.condIdx(ii,2)>=8);
        
    end
    therows = find(subs);
    if ~isempty(therows)
        for nu = 1:nunits
            u = nu;
            ctNR(1:length(therows),i,nu) = rate.count(therows,u);
            trNR(1:length(therows),i,nu) = rate.raw(therows, u);
        end
    end
end
%% (2) Add to the current dv structure 
dv.tune.trNRA = trNRA;
dv.tune.trNR = trNR;
dv.tune.trMuNR = squeeze(nanmean(trNR));
dv.tune.ctZNR = squeeze(nanmean(ctNR))./squeeze(nanstd(ctNR));

%% (3) Run the simple PID decoder 
clear logL;clear idxSELECT; clear idxRUN;
ptrain    = 0.9; % percent of training 
idxRUN    = {};  % idx training for Vel PID
idxSELECT = {};  % idx testing for Vel PID 

for run = 1:nrun
    [f,~,idxtest,idxselect,idxrun]=generateETuningNR(dv,ptrain,idxNRA,nchunks,tuning);
    idxSELECT{run} = idxselect;
    idxRUN{run} = idxrun;

    condVar;

    [indxTrial,indxAcc] = find(idxtest(:,:,1));

    for i = 1:length(indxTrial)
        for c = 1:nchunks
            for u = 1:nunits
                r(u) = dv.tune.trNRA(indxTrial(i),indxAcc(i),u,c);
            end

            for acc = 1:nAcc
                % for every testing trial, calculate the loglikelihood
                logL(i,acc,c,run) = logL_acc(f,r,acc,dv,c);

            end
        end

    end
end

%% (4) Calculate PID decoding results 
clear result;clear avelogL;clear logLval;clear estaCond; clear actualAcc;
clear decodeAcc;

% Average log-likelihood across 10 runs 
avelogL = mean(logL,4);

% Decoding results of acceleration 
for i = 1:length(indxTrial)
    for t = 1:nchunks
        [logLval,estaCond] = max(avelogL(i,:,t));
        
        result(i,t) = estaCond;
        decodeAcc(i,t) = uniAcceleration(result(i,t));
    end
end

% Real acceleration condition 
actualAcc(:,1) = uniAcceleration(indxAcc); 
 
%% (5.1) Fit the line - mean of the interval line fits as the acceleration estimates
% fit4 - without the first interval (transient)
% fit5 - includes all blocks' estimates 
% fit  - every 2 chunks' fit 
clear x; clear y; clear fit; clear fit5; clear fit4;
clear p; clear x1;

% Get the fits 
% (1) fit with ALL 10 chuncks 
 for i = 1:length(decodeAcc)
    y(i,1:nchunks) = decodeAcc(i,1:nchunks);
    x = 0.05:chunkInt:0.95;
    p = polyfit(x,y(i,:),1);
    x1 = mean(x);
    fit5(i) = polyval(p,x1);
 end 

clear x; clear y;
 % (2) fit without the transient chunck 
 for i = 1:length(decodeAcc)
    y(i,:) = decodeAcc(i,2:nchunks);
    x = chunkInt+0.05:chunkInt:0.95;
    p = polyfit(x,y(i,:),1);
    x1 = mean(x);
    fit4(i) = polyval(p,x1);  
 end 

clear x; clear y;
 % (3) fit for individual chuncks
 for i = 1:length(decodeAcc)
     for k = 1:(nchunks-1)
        y(i,:) = decodeAcc(i,k:k+1);
        x = [(k-1).*chunkInt+0.05, k.*chunkInt+0.05];
        p = polyfit(x,y(i,:),1);
        x1 = mean(x);
        fit(i,k) = polyval(p,x1);
     end    
 end 

%% (5.2) Not fitting a line - directly plotting the decoding results
clear x; clear y; clear aveALL; clear aveNT; 
% aveALL - fit all the chunks' estimates
% aveNT - fit without the transient chunk 
% take the direct estimates from dedcodeAcc
aveALL = nan(length(dv.decode.decodeAcc90),1);
aveNT = nan(length(dv.decode.decodeAcc90),1);
 % (1) average ALL chunks' direct Acc estimates  
 for i = 1:length(decodeAcc)
    y(i,:) = decodeAcc(i,1:nchunks);
    aveALL(i,1) = mean(y(i,:),2);  
 end 
 
 % (2) average non-transient chunks' direct Acc estimates
 clear y;
 for i = 1:length(decodeAcc)
    y(i,:) = decodeAcc(i,2:nchunks);
    aveNT(i,1) = mean(y(i,:),2);  
 end 

 %(3) direct decoding as the estimate 
 % take the direct estimates from dedcodeAcc
%% (8)*** SAVE decoding results back to dv structures ***
dv.decode.decodeAcc90 = decodeAcc;
dv.decode.actualAAcc90 = actualAcc;
dv.decode.fitAcc90 = fit;
dv.decode.fitNTAcc90 = fit4';
dv.decode.fitALLAcc90 = fit5';
dv.decode.aveALLAcc90 = aveALL;
dv.decode.aveNTAcc90 = aveNT;

%% (9) Statistics & Save stats - Unity line fit & MIs 
clear accActual; clear accDecoded; clear RSS; clear TSS; clear Rsquared;
clear id; clear btsR2CI; clear btsRsquared; clear AI; clear VI; clear DI; 
clear DsI; clear collapseDirtr; clear collapseDisptr;

%nunits = dv.tune.nunits;
% (9.1) R-squared - against the unity line 
% setting up 
nit = 10000;
ndraw = length(dv.decode.actualAAcc90); % draw with replacement 
id = ceil(rand(ndraw,nit)*ndraw);
CIrange = 95;

% R2 with bootstrapping 
for it = 1:nit
    
    accActual = dv.decode.actualAAcc90(id(:,it),:);
    
    for i = 1:23
        if i <= 10
            accDecoded = dv.decode.decodeAcc90(id(:,it),i);
        elseif i>10 && i<=19      
            accDecoded = dv.decode.fitAcc90(id(:,it),i-10);
        elseif i ==20
            accDecoded = dv.decode.fitALLAcc90(id(:,it),1);
        elseif i == 21
            accDecoded = dv.decode.fitNTAcc90(id(:,it),1);
        elseif i ==22
            accDecoded = dv.decode.aveALLAcc90(id(:,it),1);
        elseif i == 23
            accDecoded = dv.decode.aveNTAcc90(id(:,it),1);
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

% (9.2) MIs and Acceleration and velocity selectivity Index: (max-min)./(max+min)
    % check the distribution of AI and VI across all sessions 
for u = 1:nunits
    AI(u)=(max(dv.tune.trMu(:,u))-min(dv.tune.trMu(:,u)))./...
        (max(dv.tune.trMu(:,u))+min(dv.tune.trMu(:,u)));
    
    VI(u)=(max(dv.tune.tuneMuspd(u,:))-min(dv.tune.tuneMuspd(u,:)))./...
        (max(dv.tune.tuneMuspd(u,:))+min(dv.tune.tuneMuspd(u,:)));    
end 

% (9.3) Direction and disparity selectivity Index
    % Dir:  (pref-null)./(pref+null)
    % Disp: (max-min)./(max+min)
if exist('dvT')
dirs = dvT.tune.dirs;
disps = dvT.tune.disps;

collapseDirtr  = squeeze(mean(dvT.tune.trMu,1));
collapseDisptr = squeeze(mean(dvT.tune.trMu,2));
clear prefDir; clear nullDir;
for  u = 1:nunits
    if size(dvT.tune.ori,2) == nunits
        if sum(collapseDirtr(:,u)) >0
            prefDir(u) = (find(collapseDirtr(:,u)==max(collapseDirtr(:,u))));
            if prefDir(u) <=4
                nullDir(u) = prefDir(u) + 4;
            elseif prefDir(u) > 4
                nullDir(u) = prefDir(u) - 4;
            end
        else
            prefDir(u) =nan;
            nullDir(u) = nan;
            
        end 
    elseif size(dvT.tune.ori,2) ~= nunits
        prefDir(u) = nan;
        nullDir(u) = nan;
    end
    
end
for u = 1:nunits
    if ~isnan(prefDir(u))
    DI(u)=(collapseDirtr(prefDir(u),u)-collapseDirtr(nullDir(u),u))./...
        (collapseDirtr(prefDir(u),u)+collapseDirtr(nullDir(u),u));
    
    DsI(u)=(max(collapseDisptr(:,u))-min(collapseDisptr(:,u)))./...
        (max(collapseDisptr(:,u))+min(collapseDisptr(:,u))); 
    else
        DI(u) = nan;
        DsI(u) = nan;
    end 
end

else 
    for u = 1:nunits 
        DI(u) = nan;
        DsI(u) = nan;
        
    end 

end 

[IdxCor(:,:,1),IdxCorp(:,:,1)] = corrcoef(AI',VI');
[IdxCor(:,:,2),IdxCorp(:,:,2)] = corrcoef(AI',DI');
[IdxCor(:,:,3),IdxCorp(:,:,3)] = corrcoef(AI',DsI');

% (9.4) Add the basic stats back to dv structure 
dv.stats.dynAccbtsR2U = btsRsquared;
dv.stats.dynAccbtsmuR2U = mubtsRsquared;
dv.stats.dynAccbtsR2CIU = btsR2CI;
dv.stats.AI = AI;
dv.stats.VI = VI;
dv.stats.DI = DI;
dv.stats.DsI = DsI;
dv.stats.IdxCor = IdxCor;
dv.stats.Idxcorp = IdxCorp;
dv.stats.nunits = dv.tune.nunits;

%% (10) Best fit line from the regression models 
clear accActual; clear accDecoded; clear mdl;clear id; clear ndraw;clear am; clear vm; clear muR2;
% (10.a) - compute the bootstrapped R2 and CIs for best line of fit
%nunits = dv.tune.nunits;
nit = 1000;
ndraw = length(dv.decode.actualAAcc90); % draw with replacement 
id = ceil(rand(ndraw,nit)*ndraw);
CIrange = 95;

% fitlm with bootstrapping 
parfor it = 1:nit
    
    accActual = dv.decode.actualAAcc90(id(:,it),:);
    
    for i = 1:23
        if i <= 10
            accDecoded = dv.decode.decodeAcc90(id(:,it),i);
        elseif i>10 && i<=19      
            accDecoded = dv.decode.fitAcc90(id(:,it),i-10);
        elseif i ==20
            accDecoded = dv.decode.fitALLAcc90(id(:,it),1);
        elseif i == 21
            accDecoded = dv.decode.fitNTAcc90(id(:,it),1);
        elseif i ==22
            accDecoded = dv.decode.aveALLAcc90(id(:,it),1);
        elseif i == 23
            accDecoded = dv.decode.aveNTAcc90(id(:,it),1);
        end 
        % run the linear model 
        mdl = fitlm(accActual,accDecoded);
        am{i,it} = struct('coef',{mdl.Coefficients},'R2',{mdl.Rsquared});
       
    end   
end

for i = 1:23
    for it = 1:nit 
        AM(i,it) = am{i,it}.R2.Adjusted;
    end
end 

muR2 = nanmean(AM,2)'
% compute CIs 
for i = 1:size(AM,1)
    R2CI(i,:) = prctile(AM(i,:),[50-CIrange/2,50+CIrange/2]);
end

% (10.b) - save the basic stats back to the dv structure
dv.stats.dynAccbtsmuR2 = muR2';
dv.stats.dynAccbtsR2 = AM;
dv.stats.dynAccbtsR2CI = R2CI;

%% (11) !!!!!!! PLOTs !!!!!!!
% (a) plot the R2 distributions & CIs 
figure(666); 
ax1 = subplot(321); 
histogram(btsRsquared(10,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(10,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(10,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('1sec');
ax2 = subplot(322); 
histogram(btsRsquared(11,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(11,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(11,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('200-1000ms');
ax3 = subplot(3,5,6);
histogram(btsRsquared(1,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(1,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(1,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('0-400ms');
ax4 = subplot(3,5,7);
histogram(btsRsquared(2,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(2,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(2,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('200-600ms');
ax5 = subplot(3,5,8);
histogram(btsRsquared(3,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(3,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(3,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('400-800ms');
ax6 = subplot(3,5,9);
histogram(btsRsquared(4,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(4,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(4,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('600-1000ms');
ax7 = subplot(3,5,10);
histogram(btsRsquared(5,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(5,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(5,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('0-200ms');
ax8 = subplot(3,5,11);
histogram(btsRsquared(6,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(6,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(6,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('200-400ms');
ax9 = subplot(3,5,12);
histogram(btsRsquared(7,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(7,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(7,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('400-600ms');
ax10 = subplot(3,5,13);
histogram(btsRsquared(8,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(8,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(8,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('600-800ms');
ax11 = subplot(3,5,14);
histogram(btsRsquared(9,:)); hold on;ylim([0, 1000]);
plot(btsR2CI(9,1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(btsR2CI(9,2)*[1,1],ylim,'r--','LineWidth',2);hold on;title('800-1000ms');
linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11],'y');
sgtitle('Bootstrapped 95% CIs fo R2')

%% (b) PLOT - decoding by linear fit 
figure(777);
actualAcc = dv.decode.actualAAcc90;
ax1 = subplot(321);
scatter(actualAcc,fit5(:,:)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('full stimulus presentation - 1sec, R2 = ',num2str(mubtsRsquared(20))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax2 = subplot(322);
scatter(actualAcc,fit4(:,:)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('without transient, R2 = ',num2str(mubtsRsquared(21))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax1,ax2],'xy');

ax3 = subplot(3,5,6);
scatter(actualAcc,fit(:,1)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('0-200ms, R2 = ',num2str(mubtsRsquared(11))))
ax4 = subplot(3,5,7);
scatter(actualAcc,fit(:,2)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('100-300ms, R2 = ',num2str(mubtsRsquared(12))))
ax5 = subplot(3,5,8);
scatter(actualAcc,fit(:,3)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('200-400ms, R2 = ',num2str(mubtsRsquared(13))))
ax6 = subplot(3,5,9);
scatter(actualAcc,fit(:,4)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('300-500ms, R2 = ',num2str(mubtsRsquared(14))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
ax7 = subplot(3,5,10);
scatter(actualAcc,fit(:,5)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('400-600ms, R2 = ',num2str(mubtsRsquared(15))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
ax8 = subplot(3,5,11);
scatter(actualAcc,fit(:,6)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('500-700ms, R2 = ',num2str(mubtsRsquared(16))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
ax9 = subplot(3,5,12);
scatter(actualAcc,fit(:,7)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('600-800ms, R2 = ',num2str(mubtsRsquared(17))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
ax10 = subplot(3,5,13);
scatter(actualAcc,fit(:,8)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('700-900ms, R2 = ',num2str(mubtsRsquared(18))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
ax11 = subplot(3,5,14);
scatter(actualAcc,fit(:,9)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Estimated Acceleration (deg/s^2)');
title(strcat('800-1000ms, R2 = ',num2str(mubtsRsquared(19))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11],'xy');
sgtitle("MT - Dynamic Acceleration Decoder 90/10 - linear fit (baseline)")

%% (c) PLOT - direct decoding 
figure(888);
ax1 = subplot(321);
scatter(actualAcc,aveALL(:,:)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('full stimulus presentation - 1sec, R2 = ',num2str(muR2(22))))

ax2 = subplot(322);
scatter(actualAcc,aveNT(:,:)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('without transient, R2 = ',num2str(muR2(23))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax1,ax2],'xy');

ax3 = subplot(3,5,6);
scatter(actualAcc,decodeAcc(:,1)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('0-100ms, R2 = ',num2str(muR2(1))))

ax4 = subplot(3,5,7);
scatter(actualAcc,decodeAcc(:,2)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('100-200ms, R2 = ',num2str(muR2(2))))

ax5 = subplot(3,5,8); 
scatter(actualAcc,decodeAcc(:,3)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('200-300ms, R2 = ',num2str(muR2(3))))

ax6 = subplot(3,5,9);
scatter(actualAcc,decodeAcc(:,4)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
title(strcat('300-400ms, R2 = ',num2str(muR2(4))))

ax7 = subplot(3,5,10);
scatter(actualAcc,decodeAcc(:,5)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('400-500ms, R2 = ',num2str(muR2(5))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax3,ax4,ax5,ax6,ax7],'xy');

ax8 = subplot(3,5,11);
scatter(actualAcc,decodeAcc(:,6)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('500-600ms, R2 = ',num2str(muR2(6))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax9 = subplot(3,5,12);
scatter(actualAcc,decodeAcc(:,7)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('600-700ms, R2 = ',num2str(muR2(7))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax10 = subplot(3,5,13);
scatter(actualAcc,decodeAcc(:,8)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('700-800ms, R2 = ',num2str(muR2(8))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax11 = subplot(3,5,14);
scatter(actualAcc,decodeAcc(:,9)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('800-900ms, R2 = ',num2str(muR2(9))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');

ax12 = subplot(3,5,15);
scatter(actualAcc,decodeAcc(:,10)); hold on; 
xlabel('Actual Acceleration (deg/s^2)');
ylabel('Decoded Acceleration (deg/s^2)');
title(strcat('900-1000ms, R2 = ',num2str(muR2(10))))
ylim([-150, 150]);xlim([-150, 150]);plot(xlim,xlim);axis('square');
linkaxes([ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12],'xy');

sgtitle("MT - Dynamic Acceleration Decoder 90/10 - direct decoding")

%% (d.1) PLOT - Index plots - AI, VI, DI, DsI
% Index distributions - session 
figure(3290); clear ylim; clear xlim;
ctAI = [mean(AI),median(AI)];
ctVI = [mean(VI),median(VI)];
ctDI = [mean(DI),median(DI)];
ctDsI = [mean(DsI),median(DsI)];
subplot(221);histogram(dv.stats.AI,'BinWidth',0.05);hold on;
ylim([0 20]);xlim([0 1]);
plot(ctAI(1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(ctAI(2)*[1,1],ylim,'m--','LineWidth',2);hold on;
title('Acceleration Index');legend('AI','mean','median')
subplot(222);histogram(dv.stats.VI,'BinWidth',0.05);hold on; 
ylim([0 20]);xlim([0 1]);
plot(ctVI(1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(ctVI(2)*[1,1],ylim,'m--','LineWidth',2);hold on;
title('Velocity Index');legend('VI','mean','median')
subplot(223);histogram(dv.stats.DI,'BinWidth',0.05);hold on;
ylim([0 20]);xlim([0 1]);
plot(ctDI(1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(ctDI(2)*[1,1],ylim,'m--','LineWidth',2);hold on;
title('Direction Index');legend('DI','mean','median')
subplot(224);histogram(dv.stats.DsI,'BinWidth',0.05);hold on; 
ylim([0 20]);xlim([0 1]);
plot(ctDsI(1)*[1,1],ylim,'r--','LineWidth',2);hold on;
plot(ctDsI(2)*[1,1],ylim,'m--','LineWidth',2);hold on;
title('Disparity Index');legend('DsI','mean','median')
sgtitle('Session Index')

%% (d.2) PLOT - Paired indices for each unit
% scatter plots 
figure(); clear ylim; clear xlim;
subplot(231);scatter(dv.stats.VI,dv.stats.AI,25,"filled");hold on;
xlim =([0,1]); ylim =([0,1]);plot(xlim,xlim);axis('square');
xlabel('Velocity Index');ylabel('Acceleration Index');
subplot(232);scatter(dv.stats.DI,dv.stats.AI,25,"filled");hold on;
xlim =([0,1]); ylim =([0,1]);plot(xlim,xlim);axis('square');
xlabel('Direction Index');ylabel('Acceleration Index');
subplot(233);scatter(dv.stats.DsI,dv.stats.AI,25,"filled");hold on;
xlim =([0,1]); ylim =([0,1]);plot(xlim,xlim);axis('square');
xlabel('Disparity Index');ylabel('Acceleration Index');
subplot(2,1,2);scatter3(dv.stats.VI,dv.stats.DI,dv.stats.DsI,dv.stats.AI.*100,"filled");hold on;
xlim =([0,1]); ylim =([0,1]);axis('square'); % size is the Acceleration Index 
xlabel('Velocity Index');ylabel('Direction Index');zlabel('Disparity Index');
sgtitle('VI/DI/DsI against AI');

% AI to other index - line plots
figure(); clear ylim; clear xlim;
for u = 1:nunits
    x = [1,2]; 
    subplot(131);y = [dv.stats.AI(u),dv.stats.VI(u)]; title('AI - VI');
    plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
    subplot(132);y = [dv.stats.AI(u),dv.stats.DI(u)]; title('AI - DI');
    plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
    subplot(133);y = [dv.stats.AI(u),dv.stats.DsI(u)];title('AI - DsI');
    plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]); 
    sgtitle('AI to other Index')
end 

% Index trends relative to AI 
figure(); clear y; clear x;
subplot(211)
for u = 1:nunits
    x = [u,u,u,u]; 
    y = [dv.stats.AI(u),dv.stats.VI(u),dv.stats.DI(u),dv.stats.DsI(u)];
    plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;   
end 
subplot(212); clear y; clear x;
x = 1:1:nunits;
y = [dv.stats.AI',dv.stats.VI',dv.stats.DI',dv.stats.DsI'];
plot(x,y(:,1),'o-','LineWidth',2,'MarkerSize',10);hold on;
plot(x,y(:,2),'o-','LineWidth',2,'MarkerSize',10);hold on;
plot(x,y(:,3),'o-','LineWidth',2,'MarkerSize',10);hold on;
plot(x,y(:,4),'o-','LineWidth',2,'MarkerSize',10);hold on;
legend('AI','VI','DI','DsI'); sgtitle('unit index')

%% (e) PLOT - RF and Acc/Vm fitting quality 
figure();
x = 1:1:nunits;
y = [dv.stats.AI',dv.stats.VI',dvRF.rf.r2];
plot(x,y(:,1),'o-','LineWidth',2,'MarkerSize',10);hold on;
plot(x,y(:,2),'o-','LineWidth',2,'MarkerSize',10);hold on;
plot(x,y(:,3),'o-','LineWidth',2,'MarkerSize',10);hold on;
legend('AI','VI','RFr2'); sgtitle('unit index & RF R2')
     
%% (6) Function - Generate ensemble tuning

function [f,idxtrain,idxtest,idxselect,idxrun] = generateETuningNR(dv,ptrain,idxNRA,nchunks,tuning)

nAccelerations = size(dv.tune.tr,2);

% set up the indices for 1) NRA data structure, 2) VEL PID data structure
idxtrain = zeros([size(dv.tune.trNRA,[1,2,3])]);
idxtest = zeros([size(dv.tune.trNRA,[1,2,3])]);
idxselect = [];
idxrun = [];

% set up f - tuning functions
f = zeros([size(dv.tune.trNRA,[2,3,4])]);

% For each acceleration condition choose n% of the non-NaN trials
for aa = 1:nAccelerations

    nTrials = sum(~isnan(dv.tune.trNRA(:,aa,1)));
    nChoose = floor(ptrain*nTrials);

    % set training samples for this acceleration
    idx = randsample(nTrials,nChoose);
    idxrun = [idxrun,idxNRA{aa}(idx)];
    idxtrain(idx,aa,:) = 1;
    
    % for each unit, tuning at each Acc for each chunk
        % ** check the tuning mode 
    for c = 1:nchunks
        if ~isempty(idx)
            if tuning == 1
                f(aa,:,c) = squeeze(mean(dv.tune.trNRA(idx,aa,:,c),1));
            elseif tuning == 2
                f(aa,:,c) = squeeze(mean(dv.tune.trNRA(idx,aa,:,1),1));              
            elseif tuning == 3
                f(aa,:,c) = squeeze(mean(dv.tune.trNR(idx,aa,:),1));
                
            end 
        end
    end

    % set testing samples for this acceleration
    idx = setdiff(1:nTrials,idx);
    idxtest(idx,aa,:) = 1;
    idxselect = [idxselect,idxNRA{aa}(idx)];

end

end

%% (7) Function - Calculate the loglikelihood across nunits 

function logL = logL_acc(f,r,aCond,dv,c)

for u = 1:dv.tune.nunits
    fu(u) = f(aCond,u,c);
    logfru(u) = log(fu(u)).*r(u);
    logru(u) = logfactorial(r(u));
end

logL = sum(logfru) - sum(fu) - sum(logru);

end




