%% Solving Aiyagari Model (1994) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 10.08.2024
% Author @ Iman Taghaddosinejad (https://github.com/iman-nejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a stationary RE-RCE using VFI + Howard Improvement
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file incorporates techniques used by Hanbaek Lee in his code 
% (https://github.com/hanbaeklee/ComputationLab). All mistakes are my own.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterization
% 2. Guess price(s) or Aggregate(s)
% 3. Solve HH optimizaiton problem using VFI + Howard Improvement 
% 4. Compute staitonary distribution using non-stochastic itarative method
% 5. update price(s)/Aggregate(s)
% 6. Check convergence, if not repeat steps 3-6.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
clear 
clear all 
clc
addpath('./Functions')

%% Set Parameters 

% Aiyagari (1994) 
pAlpha = 0.36;
pBeta = 0.96;
pDelta = 0.08;
pTFP = 1;

% income process 
pMu = 0;
pRho = 0.90;
pSigma = 0.2; %0.2 * sqrt(1 - pRho^2);
pNz = 7;

% wealth grid 
pNa1 = 50;
pNa2 = 100;
pMinGrida = 0;
pMaxGrida = 150;
pCurve = 7;

% loop parameters 
pWtOldK = 0.990;
pTolGE = 1e-8;
pTolVF = 1e-8;
pTolDist = 1e-8;
pMaxGEiter = 2000;
pMaxVFiter = 20000;
convergedGE = false;
convergedVF = false;

%% Define Grids 

% wealth 
x1 = linspace(0, 0.5, pNa1);
x2 = linspace(0, 0.5, pNa2);
y1 = x1.^pCurve / max(x1.^pCurve);
y2 = x2.^pCurve / max(x2.^pCurve);
vGrida1 = pMinGrida + (pMaxGrida - pMinGrida) * y1;
vGrida2 = pMinGrida + (pMaxGrida - pMinGrida) * y2;

% productivity 
[vZ, mPz] = fnTauchen(pRho, pSigma, pMu, pNz);
vPiz = fnStationaryDist(mPz);
vZ = exp(vZ);

%% Initialize 

% initial guess (aggregate(s)) 
aggKinit = 7.8;
mVF = repmat(0.01 * vGrida1', 1, pNz);
mCurrDist = ones(pNa2, pNz);
mCurrDist = mCurrDist ./ (pNa2 * pNz);

% placeholders 
mVFnew = zeros(pNa1, pNz);
mPolaprime1 = zeros(pNa1, pNz);
mPolc = zeros(pNa1, pNz);
aggKnew = 0;
%mNewDist = zeros(pNa2, pNz);
%mPolaprime2 = zeros(pNa2, pNz);


%% Solve Model

aggK = aggKinit;
aggL = vZ' * vPiz; 
iterGE = 1;
errGE = 10;
tic; 
while errGE >= pTolGE && iterGE <= pMaxGEiter

    r = pAlpha * pTFP * (aggK / aggL)^(pAlpha - 1) - pDelta;
    w = (1 - pAlpha) * pTFP * (aggK / aggL)^pAlpha;

    % --------------------------------------------------- %
    % INNER LOOP - VFI + HOWARD IMPROVEMENT
    % --------------------------------------------------- %

    iterVF = 1;
    errVF = 10;
    while errVF >= pTolVF && iterVF <= pMaxVFiter
        
        expVF = mVF * mPz';
        
        for iz = 1:pNz
            
            expVFz = expVF(:, iz);
            minWealth = pMinGrida;
    
            for ia = 1:pNa1
                
                % ----- VFI w/ Interpolated Policy Function ----- %
        
                if iterVF <= 30 || mod(iterVF, 20) == 0
                    
                    budget = w*vZ(iz) + (1 + r)*vGrida1(ia); 
                    lb = minWealth;
                    ub = budget;
                    aprime = fnOptaprime({pBeta, budget, vGrida1, pNa1, expVFz}, lb, ub);
                    c = budget - aprime;
    
                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVFz(LB) + wtUB*expVFz(UB); % interpolate E[V]
    
                    mVFnew(ia, iz) = log(c) + pBeta * value;
                    mPolaprime1(ia, iz) = aprime;
                    mPolc(ia, iz) = c;
                    minWealth = aprime; % exploit monotonicity of policy rule
                
                % ----- VFI w/ Accelerated Howard Improvement ----- % 
                
                else
    
                    aprime = mPolaprime1(ia, iz);
                    c = mPolc(ia, iz);
                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVFz(LB) + wtUB*expVFz(UB); 
                    
                    mVFnew(ia, iz) = log(c) + pBeta * value;
    
                end
            end
        end
    
        errVF = max(max(abs(mVFnew - mVF)));
        mVF = mVFnew;
        
        %if mod(iterVF, 50) == 0
        %    fprintf('Iteration: %d, Error: %.6f\n', iterVF, errVF);
        %end        
    
        iterVF = iterVF + 1;
    end
    
    % --------------------------------------------------- %
    %  COMPUTE STATIONARY DISTRIBUTION 
    % --------------------------------------------------- %

    mPolaprime2 = interp1(vGrida1, mPolaprime1, vGrida2, 'linear', 'extrap');
    
    errDist = 10;
    iterDist = 1;
    while errDist >= pTolDist
        
        mNewDist = zeros(pNa2, pNz);

        for iz = 1:pNz
            for ia = 1:pNa1
    
                aprime = mPolaprime2(ia, iz);
                [L, H, wtL, wtH] = fnInterp1dGrid(aprime, vGrida2, pNa2);
                mass = mCurrDist(ia, iz);
    
                for iznext = 1:pNz
                    mNewDist(L, iznext) = mNewDist(L, iznext) ...
                        + mPz(iz, iznext) * mass * wtL;

                    mNewDist(H, iznext) = mNewDist(H, iznext) ...
                        + mPz(iz, iznext) * mass * wtH;
                end
            end
        end

        errDist = max(max(abs(mNewDist - mCurrDist)));
        mCurrDist = mNewDist;
        iterDist = iterDist + 1;

        %if mod(iterDist, 20) == 0
        %    fprintf('Iteration: %d, Error: %.6f\n', iterDist, errDist);
        %end
    end

    % --------------------------------------------------- %
    %  UPDATE AGGREGATES USING MARKET CLEARING 
    % --------------------------------------------------- %
    
    vMarginalDista = sum(mCurrDist, 2); % (marginal) density over asset-grid
    Kmcc = vGrida2 * vMarginalDista; % capital market clearing condition
    aggKnew = pWtOldK*aggK + (1-pWtOldK)*Kmcc; % smooth updating 

    errGE = abs(aggKnew - aggK);
    aggK = aggKnew; 

    timer = toc;
    fprintf('Iteration: %d in %.3f mins. Error: %.8f\n', iterGE, timer./60, errGE);
    fprintf('AggK: %.4f\n', aggK);
    fprintf('r: %.4f, w: %.4f\n', r, w);
    fprintf('----------------------------------------------------\n')

    iterGE = iterGE + 1;
end



plot(vGrida2, mPolaprime2(:, 1), 'o-', vGrida2, mPolaprime2(:, 7), 'x-');

plot(vGrida2, mPolalt(:, 1), 'o-', vGrida2, mPolaprime2(:, 1), 'x-');           
         


mPolaprime2 = interp1(vGrida1, mPolaprime1, vGrida2, 'linear', 'extrap');

mPolalt = zeros(pNa2, pNz);

for iz = 1:pNz 
    mPolalt(:, iz) = interp1(vGrida1, mPolaprime1(:, iz), vGrida2, "linear", "extrap");
end

mdiff = mPolalt - mPolaprime2;