import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vtk
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
import os
from readVTK import read_vtk_polydata as readVTK
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score
from scipy.optimize import minimize  # Add this import



# FILES TO READ

pigID = np.array(['ZPAF23S018', 'ZPAF23S020', 'ZPAF23S021'])
pigDate = np.array(['20230531', '20230418', '20230503'])

inletArray = np.array([7917, 8562, 5744])
bolusInletArray = np.array([9409, 7377, 5549])


pig = 'ZPAF23S018'


bloodFlowRangeArray = [[205.73352, 285.13944],[110.48652, 153.13044], [129.53592, 179.53224]]
bloodFlowRangeArray = [[205.73352, 250.0],[110.48652, 153.13044], [129.53592, 179.53224]]


bloodFlowRange = bloodFlowRangeArray[np.argwhere(pigID == pig)[0][0]]

q_array = np.arange(bloodFlowRange[0], bloodFlowRange[1] + (bloodFlowRange[1] - bloodFlowRange[0])/10 , (bloodFlowRange[1] - bloodFlowRange[0])/10)


folder =  pig + '/' + pigDate[np.argwhere(pigID == pig)[0][0]]  + '/'


# READ FILES
dataBase = pd.read_csv(folder + 'sorted_dataBase.csv')
pts = pd.read_csv(folder + 'sorted_points.csv')
dataBase = dataBase.rename(columns={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4})
pts = pts.rename(columns={'0': 0, '1': 1, '2': 2, '3': 3})

# CONVERT VALUES TO SI IF NEEDED
dataBase[[3,4]] *= 1e-3

# BOUNDARY CONDITIONS

MAP = 100 # MEan Arterial Pressure
CVP = 5 # Central Venous Pressure 

Pin =  MAP * 133.322 # 1000
Pout = CVP * 133.322 # 1

#gamma_a =  7E-15

# BLOOD PARAMETERS
myu_blood = 8.9e-4 # Pa-s
rho_blood = 1045 # Kg/m3
Cp_blood = 3600 # J/KgK

# BOLUS PARAMETERS
myu_bolus = 7E-4 # Pa-s
rho_bolus = 1280 # Kg/m3
Cp_bolus = 1970 # J/KgK

# liverPerfusion = 179.53224


# IDENTIFY INLETS AND OUTLETS
combined_values = pd.concat([dataBase[1], dataBase[2]])
common_values_counts = combined_values.value_counts()
endPoints = list(set(common_values_counts[common_values_counts == 1].index))#.sort
endPoints.sort()

# FLOW INLETS
acceptableOutlets = np.load(folder + 'acceptable_Outlets.npy')



inlets = inletArray[np.argwhere(pigID == pig)[0]]



outlets = [value for value in acceptableOutlets if value not in inlets]
stubs = [point for point in endPoints if point not in inlets and point not in outlets]

# contrastInlets = [5549]
# CLEAR MEMORY
del(combined_values, common_values_counts, endPoints)

# DCACI INLETS
# contrastInlets = [5549]

radiusWeightage = [1] 
radiusWeightage = radiusWeightage / np.sum(radiusWeightage)





dV = dataBase.iloc[:, 3].values * np.pi * (dataBase.iloc[:, 4].values) ** 2
dA_surf = 2 * np.pi * dataBase.iloc[:,4].values * dataBase.iloc[:,3].values
Acs = np.pi * dataBase.iloc[:,4].values ** 2

kart = np.pi * dataBase.iloc[:, 4].values**4 / (8 * myu_blood * dataBase.iloc[:, 3].values)

def flowSolver(kart, startEle, stopEle, gamma_a):
    
    row = []
    col = []
    data = []
    
    for ele in range(startEle, stopEle):
        n1, n2 = np.array(dataBase.iloc[ele, 1:3].values, dtype = int)
        flowCond = kart[ele]

        if n1 not in inlets:
            row.extend([n1, n1])
            col.extend([n1, n2])
            data.extend([-flowCond, flowCond])
        
        if n2 not in inlets:
            row.extend([n2, n2])
            col.extend([n2, n1])
            data.extend([-flowCond, flowCond])
        
        if n1 in inlets:
            row.append(n1)
            col.append(n1)
            data.append(1)
        
        if n2 in inlets:
            row.append(n2)
            col.append(n2)
            data.append(1)
        
        if n1 in outlets:
            row.append(n1)
            col.append(n1)
            data.append( -gamma_a/myu_blood)
        
        if n2 in outlets:
            row.append(n2)
            col.append(n2)
            data.append( -gamma_a/myu_blood)
    
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    A = sp.coo_matrix((data, (row,col)), shape = (nNodes, nNodes))
    b = np.zeros((nNodes), dtype = float)
    
    for ii in inlets:
        b[int(ii)] = Pin
    for ii in outlets:
        b[int(ii)] = -gamma_a/myu_blood * Pout
    
    P_art = spla.spsolve(A,b)
    
    # CALCULATE FLOW ERRORS
    # error = A.dot(P_art) - b
    n1 = dataBase.iloc[:, 1].astype(int).values
    n2 = dataBase.iloc[:, 2].astype(int).values
    Qa = kart * (P_art[n1] - P_art[n2])
    # print('Residual Error = ', np.linalg.norm(error))
    
    outEle = dataBase[dataBase[1].isin(outlets) | dataBase[2].isin(outlets)]
    inEle = dataBase[dataBase[1].isin(inlets) | dataBase[2].isin(inlets)]
    
    # outFlow = np.sum(np.abs(Qa[outEle.iloc[:, 0].values]))
    inFlow = np.sum(np.abs(Qa[inEle.iloc[:, 0].values]))
    # errorFlow = inFlow - outFlow        
    # print('Flow error = ', errorFlow)
    # print('Flow = ', inFlow * 6e7)
    
    perfusionError = np.abs(liverPerfusion - inFlow * 6e7) ** 2
    print('\nperfusion Error = ', perfusionError, liverPerfusion, inFlow * 6e7, gamma_a )
    
    return perfusionError 




def objective_function(params):
    gamma_a = params[0] 
    # Run the simulation
    perfusionError = flowSolver(kart, 0, nEle, gamma_a)
    
    # print(gamma_a, perfusionError)

    return perfusionError
    # return total_error



if __name__=='__main__':
    
    nNodes = np.max(dataBase.iloc[:, 1:3].values) + 1
    nEle = len(dataBase)
    # print('Number of elements = ', nEle)
    # print('Number of nodes = ', nNodes)
    ref = []
    for liverPerfusion in q_array:
    
        initial_guess = [2E-15]
        bounds = [(1E-15, 5E-5)]
        # Run the optimization
        # result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
        result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
        # Extract optimized parameters
        optimal_gamma = result.x
        
        print('\nliverPerfusion = ', liverPerfusion)
        print('\noptimal gamma = ', optimal_gamma)
        
        # print(f"Optimal parameters: dt1={optimal_dt1}, dt2={optimal_dt2}, myuC={optimal_myuC}, tC={optimal_tC}")
        # print(f"Optimal error: {result.fun}")
        
        
        x = flowSolver(kart, 0, nEle, optimal_gamma[0])   
        
        ref.append([liverPerfusion, optimal_gamma[0]])
    
    ref = pd.DataFrame(ref)
    ref.to_csv(pig + 'optimumGa.csv', index=False)
    
    