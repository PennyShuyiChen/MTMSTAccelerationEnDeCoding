%% Plotting the boostrapped r2
% LAST UPDATE - for MT datasets, 06/12/2024 PSC 


%% Load & Calculate the bts averages
% All, Kipp, Leo 
% (1) MT 
load("DynBr2s.mat")
aveAcc  = [nanmean(br2Acc,1);nanmean(br2Acc(1:12,:),1);nanmean(br2Acc(13:21,:),1)];
aveAccT = [nanmean(br2AccT,1);nanmean(br2AccT(1:12,:),1);nanmean(br2AccT(13:21,:),1)];
aveAccF = [nanmean(br2AccF,1);nanmean(br2AccF(1:12,:),1);nanmean(br2AccF(13:21,:),1)];

aveVm  = [nanmean(br2Vm,1);nanmean(br2Vm(1:12,:),1);nanmean(br2Vm(13:21,:),1)];
aveVmT = [nanmean(br2VmT,1);nanmean(br2VmT(1:12,:),1);nanmean(br2VmT(13:21,:),1)];
aveVmF = [nanmean(br2VmF,1);nanmean(br2VmF(1:12,:),1);nanmean(br2VmF(13:21,:),1)];

% (2) MST 
load("DynBr2sMST.mat")
aveAccS  = [nanmean(br2AccS,1);nanmean(br2AccS(1:9,:),1);nanmean(br2AccS(10:11,:),1)];
aveAccTS = [nanmean(br2AccTS,1);nanmean(br2AccTS(1:9,:),1);nanmean(br2AccTS(10:11,:),1)];
aveAccFS = [nanmean(br2AccFS,1);nanmean(br2AccFS(1:9,:),1);nanmean(br2AccFS(10:11,:),1)];

aveVmS  = [nanmean(br2VmS,1);nanmean(br2VmS(1:9,:),1);nanmean(br2VmS(10:11,:),1)];
aveVmTS = [nanmean(br2VmTS,1);nanmean(br2VmTS(1:9,:),1);nanmean(br2VmTS(10:11,:),1)];
aveVmFS = [nanmean(br2VmFS,1);nanmean(br2VmFS(1:9,:),1);nanmean(br2VmFS(10:11,:),1)];

%% MT plots 
% (1) Acceleration Direct Decoding - 60ms latency, 100ms chunk 
figure(); clear ylim; clear xlim;
% (a) Dynamic Acceleration Tuning 
subplot(321);x = [1,2];y = aveAcc(1,22:23); yk = aveAcc(2,22:23);
yl = aveAcc(3,22:23);title('With & Without transient');
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2Acc,1)
   ys = br2Acc(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - with & Without transient')
subplot(322);x = (1:1:10);y = aveAcc(1,1:10); yk = aveAcc(2,1:10);
yl = aveAcc(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2Acc,1)
   ys = br2Acc(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - latency 60ms, block 100ms');

% (b) Transient Tuning 
subplot(323);x = [1,2];y = aveAccT(1,22:23); yk = aveAccT(2,22:23);
yl = aveAccT(3,22:23);title('With & Without transient');
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccT,1)
   ys = br2AccT(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - with & Without transient')
subplot(324);x = (1:1:10);y = aveAccT(1,1:10); yk = aveAccT(2,1:10);
yl = aveAccT(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccT,1)
   ys = br2AccT(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - latency 60ms, block 100ms');

% (c) Full trial tuning 
subplot(325);x = [1,2];y = aveAccF(1,22:23); yk = aveAccF(2,22:23);
yl = aveAccF(3,22:23);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccF,1)
   ys = br2AccF(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - with & Without transient')
subplot(326);x = (1:1:10);y = aveAccF(1,1:10); yk = aveAccF(2,1:10);
yl = aveAccF(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccF,1)
   ys = br2AccF(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - latency 60ms, block 100ms');
sgtitle('Acceleration Direct Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');


% (2) Acceleration & Vm Fit Decoding - 60ms latency, 200ms intervals 
figure(); clear ylim; clear xlim;
% (a) Dynamic Tuning 
subplot(321);x = (1:1:9);y = aveAcc(1,11:19); yk = aveAcc(2,11:19);
yl = aveAcc(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2Acc,1)
   ys = br2Acc(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - 200ms interval');

subplot(322);x = (1:1:9);y = aveVm(1,1:9); yk = aveVm(2,1:9);
yl = aveVm(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2Vm,1)
   ys = br2Vm(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Vm Tuning - 200ms interval');

% (b) Transient Tuning 
subplot(323);x = (1:1:9);y = aveAccT(1,11:19); yk = aveAccT(2,11:19);
yl = aveAccT(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccT,1)
   ys = br2AccT(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([-0.3,1]);hold on;
end
title('Transient Acc Tuning - 200ms interval');

subplot(324);x = (1:1:9);y = aveVmT(1,1:9); yk = aveVmT(2,1:9);
yl = aveVmT(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2VmT,1)
   ys = br2VmT(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Transient Vm Tuning - 200ms interval');

% (c) Full trial tuning 
subplot(325);x = (1:1:9);y = aveAccF(1,11:19); yk = aveAccF(2,11:19);
yl = aveAccF(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccF,1)
   ys = br2AccF(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - 200ms interval');

subplot(326);x = (1:1:9);y = aveVmF(1,1:9); yk = aveVmF(2,1:9);
yl = aveVmF(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2VmF,1)
   ys = br2VmF(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Vm Tuning - 200ms interval');
sgtitle('Acceleration & Vm Fits Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');

% (3) Acceleration & Vm Fit Decoding - with/without transient block 
figure(); clear ylim; clear xlim;
% (a) Dynamic Acceleration Tuning 
subplot(321);x = [1,2];y = aveAcc(1,20:21); yk = aveAcc(2,20:21);
yl = aveAcc(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2Acc,1)
   ys = br2Acc(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - with & Without transient')
subplot(322);x = [1,2];y = aveVm(1,10:11); yk = aveVm(2,10:11);
yl = aveVm(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2Vm,1)
   ys = br2Vm(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Vm Tuning - with & Without transient')

% (b) Transient Tuning 
subplot(323);x = [1,2];y = aveAccT(1,20:21); yk = aveAccT(2,20:21);
yl = aveAccT(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccT,1)
   ys = br2AccT(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - with & Without transient')
subplot(324);x = [1,2];y = aveVmT(1,10:11); yk = aveVmT(2,10:11);
yl = aveVmT(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2VmT,1)
   ys = br2VmT(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Vm Tuning - with & Without transient')

% (c) Full trial tuning 
subplot(325);x = [1,2];y = aveAccF(1,20:21); yk = aveAccF(2,20:21);
yl = aveAccF(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccF,1)
   ys = br2AccF(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - with & Without transient')
subplot(326);x = [1,2];y = aveVmF(1,10:11); yk = aveVmF(2,10:11);
yl = aveVmF(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2VmF,1)
   ys = br2VmF(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Vm Tuning - with & Without transient')
sgtitle('Acceleration & Vm Fits Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');


%% MST plots 
% (1) Acceleration Direct Decoding - 60ms latency, 100ms chunk 
figure(); clear ylim; clear xlim;
% (a) Dynamic Acceleration Tuning 
subplot(321);x = [1,2];y = aveAccS(1,22:23); yk = aveAccS(2,22:23);
yl = aveAccS(3,22:23);title('With & Without transient');
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccS,1)
   ys = br2AccS(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - with & Without transient')
subplot(322);x = (1:1:10);y = aveAccS(1,1:10); yk = aveAccS(2,1:10);
yl = aveAccS(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccS,1)
   ys = br2AccS(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - latency 60ms, block 100ms');

% (b) Transient Tuning 
subplot(323);x = [1,2];y = aveAccTS(1,22:23); yk = aveAccTS(2,22:23);
yl = aveAccTS(3,22:23);title('With & Without transient');
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccTS,1)
   ys = br2AccTS(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - with & Without transient')
subplot(324);x = (1:1:10);y = aveAccTS(1,1:10); yk = aveAccTS(2,1:10);
yl = aveAccTS(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccTS,1)
   ys = br2AccTS(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - latency 60ms, block 100ms');

% (c) Full trial tuning 
subplot(325);x = [1,2];y = aveAccFS(1,22:23); yk = aveAccFS(2,22:23);
yl = aveAccFS(3,22:23);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccFS,1)
   ys = br2AccFS(s,22:23);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - with & Without transient')
subplot(326);x = (1:1:10);y = aveAccFS(1,1:10); yk = aveAccFS(2,1:10);
yl = aveAccFS(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccFS,1)
   ys = br2AccFS(s,1:10);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - latency 60ms, block 100ms');
sgtitle('MST Acceleration Direct Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');


% (2) Acceleration & Vm Fit Decoding - 60ms latency, 200ms intervals 
figure(); clear ylim; clear xlim;
% (a) Dynamic Tuning 
subplot(321);x = (1:1:9);y = aveAccS(1,11:19); yk = aveAccS(2,11:19);
yl = aveAccS(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccS,1)
   ys = br2AccS(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - 200ms interval');

subplot(322);x = (1:1:9);y = aveVmS(1,1:9); yk = aveVmS(2,1:9);
yl = aveVmS(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2VmS,1)
   ys = br2VmS(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Dynamic Vm Tuning - 200ms interval');

% (b) Transient Tuning 
subplot(323);x = (1:1:9);y = aveAccTS(1,11:19); yk = aveAccTS(2,11:19);
yl = aveAccTS(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccTS,1)
   ys = br2AccTS(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([-0.3,1]);hold on;
end
title('Transient Acc Tuning - 200ms interval');

subplot(324);x = (1:1:9);y = aveVmTS(1,1:9); yk = aveVmTS(2,1:9);
yl = aveVmTS(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2VmTS,1)
   ys = br2VmTS(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Transient Vm Tuning - 200ms interval');

% (c) Full trial tuning 
subplot(325);x = (1:1:9);y = aveAccFS(1,11:19); yk = aveAccFS(2,11:19);
yl = aveAccFS(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2AccFS,1)
   ys = br2AccFS(s,11:19);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - 200ms interval');

subplot(326);x = (1:1:9);y = aveVmFS(1,1:9); yk = aveVmFS(2,1:9);
yl = aveVmFS(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
for s = 1:size(br2VmFS,1)
   ys = br2VmFS(s,1:9);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);hold on;
   ylim([0,1]);hold on;
end
title('Full Vm Tuning - 200ms interval');
sgtitle('MST Acceleration & Vm Fits Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');

% (3) Acceleration & Vm Fit Decoding - with/without transient block 
figure(); clear ylim; clear xlim;
% (a) Dynamic Acceleration Tuning 
subplot(321);x = [1,2];y = aveAccS(1,20:21); yk = aveAccS(2,20:21);
yl = aveAccS(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccS,1)
   ys = br2AccS(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Acc Tuning - with & Without transient')
subplot(322);x = [1,2];y = aveVmS(1,10:11); yk = aveVmS(2,10:11);
yl = aveVmS(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2VmS,1)
   ys = br2VmS(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Dynamic Vm Tuning - with & Without transient')

% (b) Transient Tuning 
subplot(323);x = [1,2];y = aveAccTS(1,20:21); yk = aveAccTS(2,20:21);
yl = aveAccTS(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccTS,1)
   ys = br2AccTS(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Acc Tuning - with & Without transient')
subplot(324);x = [1,2];y = aveVmTS(1,10:11); yk = aveVmTS(2,10:11);
yl = aveVmTS(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2VmTS,1)
   ys = br2VmTS(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Transient Vm Tuning - with & Without transient')

% (c) Full trial tuning 
subplot(325);x = [1,2];y = aveAccFS(1,20:21); yk = aveAccFS(2,20:21);
yl = aveAccFS(3,20:21);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2AccFS,1)
   ys = br2AccFS(s,20:21);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Acc Tuning - with & Without transient')
subplot(326);x = [1,2];y = aveVmFS(1,10:11); yk = aveVmFS(2,10:11);
yl = aveVmFS(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks([1,2]);
for s = 1:size(br2VmFS,1)
   ys = br2VmFS(s,10:11);
   plot(x,ys,'o-','LineWidth',1,'MarkerSize',5,'Color',[0,0,0,0.2]);
   ylim([0,1]);hold on;
end
title('Full Vm Tuning - with & Without transient')
sgtitle('MST Acceleration & Vm Fits Decoding - Best fits'); 
legend('All','Monkey K','Monkey L','Session');


%% (4) Look at Leo separately 
figure();
% (a) with & without transient
subplot(321);x = [1,2];y = aveAccS(3,22:23); yk = aveAccTS(3,22:23);
yl = aveAccFS(3,22:23);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
ylim([0,0.05]);title('Acc- with & Without transient')
subplot(322);x = [1,2];y = aveVmS(3,10:11); yk = aveVmTS(3,10:11);
yl = aveVmFS(3,10:11);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
ylim([0,0.05]);
title('Vm - With & Without transient')

% (b) Acc & Vm fit decoding 
subplot(323);x = (1:1:9);y = aveAccS(3,11:19); yk = aveAccTS(3,11:19);
yl = aveAccFS(3,11:19);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
title('Acc Fit Decoding')
subplot(324);x = (1:1:9);y = aveVmS(3,1:9); yk = aveVmTS(3,1:9);
yl = aveVmFS(3,1:9);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
ylim([0,0.05]);
title('Vm Fit Decoding')

% (c) Acc Direct Decoding 
subplot(325);x = (1:1:10);y = aveAccS(3,1:10); yk = aveAccTS(3,1:10);
yl = aveAccFS(3,1:10);
plot(x,y,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yk,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
plot(x,yl,'o-','LineWidth',2,'MarkerSize',10);hold on;xticks(x);
ylim([0,0.05]);
title('Acc Direct Decoding')


sgtitle('Leo MST Acceleration & Vm Fits Decoding - Best fits'); 
legend('Dynamic','Transient','Full');



