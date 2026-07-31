%% Plot_paper_figs.m -- 4 figures for the c* ML paper (transparent PNG via export_fig)
% Reads fig_data/*.csv (produced by fig_prep.py). House style: Times, publication sizes.
clear; clc; close all;
here = fileparts(mfilename('fullpath')); cd(here);

% --- function library (export_fig, distinguishable_colors) ---
cands = {'C:\Users\peifen\OneDrive\Documents\Matlab_functions','E:\Matlab_functions', ...
         'C:\Users\peifen\Documents\Matlab_functions'};
for k=1:numel(cands); if isfolder(cands{k}); addpath(genpath(cands{k})); end; end
haveEF = exist('export_fig','file')==2;

set(0,'defaultAxesFontName','Times New Roman','defaultTextFontName','Times New Roman', ...
      'defaultAxesFontSize',12,'defaultTextFontSize',12);
outdir = fullfile(here,'figures'); if ~isfolder(outdir); mkdir(outdir); end

M = readtable(fullfile(here,'fig_data','materials.csv'),'TextType','string');
imp = readtable(fullfile(here,'fig_data','importance.csv'),'TextType','string');
abl = readtable(fullfile(here,'fig_data','ablation.csv'),'TextType','string');
ps  = readtable(fullfile(here,'fig_data','parity_stats.csv'));

% anion -> color map (chalcogen/halogen families)
anions = unique(M.anion);
try; cmap = distinguishable_colors(numel(anions)); catch; cmap = lines(numel(anions)); end
acolor = @(a) cmap(find(anions==a,1),:);

%% ===== Figure 1: dataset overview =====
f1 = figure('Color','w','Position',[100 100 900 360]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
% (a) c* histogram, dense vs tail
nexttile; hold on; box on;
edges = 0.9:0.05:2.0;
histogram(M.c_star(M.regime=="dense"),edges,'FaceColor',[0.30 0.50 0.75],'FaceAlpha',0.85,'EdgeColor','w');
histogram(M.c_star(M.regime=="tail"), edges,'FaceColor',[0.85 0.45 0.35],'FaceAlpha',0.85,'EdgeColor','w');
xline(1.6,'--','Color',[0.4 0.4 0.4],'LineWidth',1);
xlabel('optimal parameter  c*'); ylabel('count');
legend({'dense (c*<1.6)','tail (c*\geq1.6)'},'Location','northeast','Box','off');
title('(a)  c* distribution','FontWeight','normal');
% (b) PBE gap vs c*, colored by anion FAMILY (grouped for a clean legend)
nexttile; hold on; box on;
halide={'F','Cl','Br','I'}; chal={'O','S','Se','Te'}; pnic={'N','P','As','Sb'}; tetrel={'C','Si','Ge'};
fam = strings(height(M),1);
for r=1:height(M)
    a = char(M.anion(r));
    if     ismember(a,halide),  fam(r)="halide";
    elseif ismember(a,chal),    fam(r)="chalcogenide";
    elseif ismember(a,pnic),    fam(r)="pnictide";
    elseif ismember(a,tetrel),  fam(r)="tetrel";
    else,                       fam(r)="other"; end
end
famNames = ["halide","chalcogenide","pnictide","tetrel","other"];
famCol   = [0.20 0.55 0.80; 0.85 0.45 0.35; 0.35 0.65 0.40; 0.60 0.45 0.70; 0.55 0.55 0.55];
hleg=[]; lname={};
for i=1:numel(famNames)
    s = fam==famNames(i);
    if any(s)
        h = scatter(M.pbe_gap(s), M.c_star(s), 42, famCol(i,:),'filled', ...
            'MarkerEdgeColor','k','MarkerFaceAlpha',0.85,'LineWidth',0.3);
        hleg(end+1)=h; lname{end+1}=char(famNames(i)); %#ok<AGROW>
    end
end
xlabel('PBE gap  (eV)'); ylabel('optimal parameter  c*');
legend(hleg,lname,'Location','southeast','Box','off','FontSize',10);
title('(b)  c* vs PBE gap by anion family','FontWeight','normal');
export_one(f1, fullfile(outdir,'Figure1_dataset.png'), haveEF);

%% ===== Figure 2: predicted vs true c* (RF, LOO) =====
f2 = figure('Color','w','Position',[100 100 460 430]); hold on; box on;
lo=1.0; hi=2.0; plot([lo hi],[lo hi],'k--','LineWidth',1);
sd = M.regime=="dense"; st = M.regime=="tail";
scatter(M.c_star(sd),M.pred_cstar(sd),40,[0.30 0.50 0.75],'filled','MarkerEdgeColor','k','MarkerFaceAlpha',0.85);
scatter(M.c_star(st),M.pred_cstar(st),40,[0.85 0.45 0.35],'filled','MarkerEdgeColor','k','MarkerFaceAlpha',0.85);
axis([lo hi lo hi]); axis square;
xlabel('true c*  (2-probe fit)'); ylabel('predicted c*  (RF, LOO)');
text(1.03,1.92,sprintf('MAE = %.3f\nR^2 = %.2f\nn = %d',ps.mae,ps.r2,ps.n),'FontSize',12,'VerticalAlignment','top');
legend({'y = x','dense','tail'},'Location','southeast','Box','off');
title('Leave-one-out prediction of c*','FontWeight','normal');
export_one(f2, fullfile(outdir,'Figure2_parity.png'), haveEF);

%% ===== Figure 3: feature importance + physical trends =====
f3 = figure('Color','w','Position',[100 100 1000 360]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
% (a) RF importance (sorted ascending in file -> barh reads bottom-up)
nexttile;
names = replace(imp.feature,'_','\_');
barh(imp.importance,'FaceColor',[0.45 0.55 0.70],'EdgeColor','none');
set(gca,'YTick',1:height(imp),'YTickLabel',names,'FontSize',9); ylim([0.4 height(imp)+0.6]);
xlabel('RF importance'); title('(a)  feature importance','FontWeight','normal');
% (b) c* vs electronegativity spread
nexttile; hold on; box on;
scatter(M.electroneg_spread,M.c_star,32,[0.30 0.50 0.75],'filled','MarkerEdgeColor','k','MarkerFaceAlpha',0.8);
p=polyfit(M.electroneg_spread,M.c_star,1); xx=linspace(min(M.electroneg_spread),max(M.electroneg_spread),20);
plot(xx,polyval(p,xx),'k-','LineWidth',1.2);
r=corr(M.electroneg_spread,M.c_star);
xlabel('electronegativity spread'); ylabel('c*');
title(sprintf('(b)  ionicity  (r = %.2f)',r),'FontWeight','normal');
% (c) c* vs band center
nexttile; hold on; box on;
scatter(M.band_center,M.c_star,32,[0.60 0.40 0.65],'filled','MarkerEdgeColor','k','MarkerFaceAlpha',0.8);
xlabel('anion-p band center  (eV)'); ylabel('c*');
title('(c)  band-center descriptor','FontWeight','normal');
export_one(f3, fullfile(outdir,'Figure3_importance.png'), haveEF);

%% ===== Figure 4: band-center resolves chalcopyrites =====
f4 = figure('Color','w','Position',[100 100 900 360]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
C = M(M.is_chalco==1,:);
% (a) within-chalcopyrite band_center vs c*, colored by anion
nexttile; hold on; box on;
ca = unique(C.anion);
try; ccol=distinguishable_colors(numel(ca)); catch; ccol=lines(numel(ca)); end
for i=1:numel(ca)
    s=C.anion==ca(i);
    scatter(C.band_center(s),C.c_star(s),50,ccol(i,:),'filled','MarkerEdgeColor','k');
end
p=polyfit(C.band_center,C.c_star,1); xx=linspace(min(C.band_center),max(C.band_center),20);
plot(xx,polyval(p,xx),'k-','LineWidth',1.2);
r=corr(C.band_center,C.c_star);
xlabel('anion-p band center  (eV)'); ylabel('c*  (chalcopyrites)');
legend(cellstr(ca),'Location','northeast','Box','off');
title(sprintf('(a)  within-family  (r = %.2f)',r),'FontWeight','normal');
% (b) ablation MAE bars: chalcopyrite vs non-chalco, with/without band_center
nexttile; box on;
grp = abl.group;
vals = [abl.mae_no_bandcenter, abl.mae_with_bandcenter];
b=bar(vals,'grouped'); b(1).FaceColor=[0.80 0.55 0.45]; b(2).FaceColor=[0.35 0.60 0.55];
set(gca,'XTickLabel',replace(cellstr(grp),'_','\_'));
ylabel('LOO MAE (c*)'); legend({'14 feat (no band center)','15 feat (+ band center)'},'Location','northoutside','Box','off','Orientation','horizontal','FontSize',9);
title('(b)  band-center ablation','FontWeight','normal');
export_one(f4, fullfile(outdir,'Figure4_bandcenter.png'), haveEF);

fprintf('\nDONE. 4 figures -> %s  (export_fig=%d)\n', outdir, haveEF);

%% ---- local export helper: transparent PNG via export_fig gcf ----
function export_one(fh, fname, haveEF)
    figure(fh); drawnow;
    ax = findall(fh,'Type','axes');          % gca(s): make plot area transparent too
    if haveEF
        set(fh,'Color','none');              % gcf = none
        set(ax,'Color','none');              % gca = none
        export_fig(fh, fname, '-transparent', '-png', '-r300');
        set(fh,'Color','w');
    else
        set(fh,'Color','none'); set(ax,'Color','none');
        set(fh,'InvertHardcopy','off'); print(fh, fname, '-dpng','-r300');
    end
    fprintf('  wrote %s\n', fname);
end
