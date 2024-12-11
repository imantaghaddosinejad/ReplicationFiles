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
pSigma = 0.2 * sqrt(1 - pRho^2);
pNz = 7;

% wealth grid 
pNa1 = 50;
pNa2 = 100;
pMinGrida = 0;
pMaxGrida = 150;
pCurve = 7;

% loop parameters 
pOldK = 0.990;
pTolGE = 1e-8;
pTolVF = 1e-8;
pTolDist = 1e-8;
pMaxGEiter = 2000;
pMaxVFiter = 2000;
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
aggKinit = 6;
mVF = repmat(0.01 * vGrida1', 1, pNz);
mCurrDist = ones(pNa2, pNz);
mCurrDist = mCurrDist ./ (pNa2 * pNz);

% placeholders 
mPolaprime1 = zeros(pNa1, pNz);
%mPolaprime2 = zeros(pNa2, pNz);
mPolc = zeros(pNa1, pNz);

aggK = aggKinit;
aggL = vZ' * vPiz; 
iterGE = 1;
errGE = 10;
while errGE >= pTolGE && iterGE <= pMaxGEiter

    r = pAlpha * pTFP * (aggK / aggL)^(pAlpha - 1) - pDelta;
    w = (1 - pAlpha) * pTFP * (aggK / aggL)^pAlpha;

    % --------------------------------------------------- %
    % INNER LOOP - VFI + Howard Improvement
    % --------------------------------------------------- %
    
    iterVF = 1;
    errVF = 10;
    while errVF >= pTolVF && iterVF <= pMaxVFiter
        
        expVF = mVF * mPz';
        
        for iz = 1:pNz
            
            expVF = expVF(:, iz);
            minWealth = pMinGrida;

            for ia = 1:pNa1
                
                % ----- VFI w/ Interpolated Policy Function ----- %
        
                if iterVF <= 30 || mod(iterVF, 20) == 0
                    
                    budget = w*vZ(iz) + (1 + r)*vGrida1(ia); 
                    lb = minWealth;
                    ub = budget;
                    aprime = fnOptaprime({pBeta, budget, vGrida1, pNa1, expVF}, lb, ub);
                    c = budget - aprime;

                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVF(LB) + wtUB*expVF(UB); % interpolate E[V]

                    mVF(ia, iz) = log(c) + pBeta * value;
                    mPolaprime1(ia, iz) = aprime;
                    mPolc(ia, iz) = c;
                    minWealth = aprime; % exploit monotonicity of policy rule
                
                % ----- VFI w/ Accelerated Howard Improvement ----- % 
                
                else

                    aprime = mPolaprime1(ia, iz);
                    c = mPolc(ia, iz);
                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVF(LB) + wtUB*expVF(UB); 
                    
                    mVF(ia, iz) = log(c) + pBeta * value;

                end

            end
        end
    end



            


           
         
            
        
           
            end









    
end


