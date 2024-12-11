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

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score


pigID = np.array(['ZPAF23S018', 'ZPAF23S020', 'ZPAF23S021'])
pigDate = np.array(['20230531', '20230418', '20230503'])

inletArray = np.array([7917, 8562, 5744])
bolusInletArray = np.array([9409, 7377, 5549])

bolusInjectedArray = [200, 250, 400]

# FILES TO READ

pig = 'ZPAF23S021'
folder = pig + '/' + pigDate[np.argwhere(pigID == pig)[0][0]]  + '/'

# READ FILES
dataBase = pd.read_csv(folder + 'sorted_dataBase.csv')
pts = pd.read_csv(folder + 'sorted_points.csv')
dataBase = dataBase.rename(columns={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4})
pts = pts.rename(columns={'0': 0, '1': 1, '2': 2, '3': 3})

# CONVERT VALUES TO SI IF NEEDED

dataBase[[3,4]] *= 1E-3  ################################### CHECK THIS BEFORE ZPAF

# BOUNDARY CONDITIONS
MAP = 100 # MEan Arterial Pressure
CVP = 5 # Central Venous Pressure 

Pin =  MAP * 133.322 # 1000
Pout = CVP * 133.322 # 1

# dataBase.iloc[9,3] = 0.005


steady = '_dmgTh_'
# reference = pd.read_csv(pig + steady + 'ref.csv')


gamma_a = 5.28E-15
timeConstant = 13.943

damageThreshold = 0.01



myu_constant = 0 

inlets = inletArray[np.argwhere(pigID == pig)[0]]
contrastInlets = bolusInletArray[np.argwhere(pigID == pig)[0]]

injectedBolus = bolusInjectedArray[np.argwhere(pigID == pig)[0][0]] * 1e-9 # m3/sec
#q_inj = 1219.37E-9 #140.37E-9 #252.489E-9 # 117.93E-9 # m3/sec


dt = 0.01 # TIME STEP FOR SOLVER


# BLOOD PARAMETERS
myu_blood = 8.9e-4 # Pa-s
rho_blood = 1045 # Kg/m3
Cp_blood = 3600 # J/KgK

# BOLUS PARAMETERS
myu_bolus = 7E-4 # Pa-s
rho_bolus = 1280 # Kg/m3
Cp_bolus = 1970 # J/KgK

# DCACI PARAMETERS
epsilon = 0.1919  # Volume fraction of DCACI to bolus
rho_DCACI = 1532 #Kg/m3


# IDENTIFY INLETS AND OUTLETS
combined_values = pd.concat([dataBase[1], dataBase[2]])
common_values_counts = combined_values.value_counts()
endPoints = list(set(common_values_counts[common_values_counts == 1].index))#.sort
endPoints.sort()

# FLOW INLETS
acceptableOutlets = np.load(folder + 'acceptable_Outlets.npy')
outlets = [value for value in acceptableOutlets if value not in inlets]
stubs = [point for point in endPoints if point not in inlets and point not in outlets]

# CLEAR MEMORY
del(combined_values, common_values_counts, endPoints)


# SIMPLIFY CALCULATIONS
convRate_constant = rho_DCACI * epsilon 


# vtkFilename = 'tc_' + str(timeConstant) + '_myuC_' + str(myu_constant) + '_stepInjection_P_in_' + str(MAP) + 'damageConvRate_soln.'

dV = dataBase.iloc[:, 3].values * np.pi * (dataBase.iloc[:, 4].values) ** 2
dA_surf = 2 * np.pi * dataBase.iloc[:,4].values * dataBase.iloc[:,3].values
Acs = np.pi * dataBase.iloc[:,4].values ** 2

refDamage, ref_pts = readVTK(folder + 'sortedVesselNetwork.vtk')
refDamage = np.array(refDamage)

def calculate_injection(dt1, dt2, max_Time, dt, q_ref, q, sat_per_second):
    if dt1 >= dt2:
        return None, None, None  # Invalid combination

    timeSteps = np.arange(0, max_Time, dt)

    def calculate_total_injection(X):
        alternating_condition = (timeSteps % (dt1 + dt2)) < dt1
        time_limit_condition = timeSteps < X
        condition = alternating_condition & time_limit_condition
        injectionCriteria = condition.astype(int)
        return np.sum(q * sat_per_second * injectionCriteria * dt)

    # Binary search to find the optimal X
    X_low, X_high = 0, max_Time
    tolerance = 0.001

    while X_high - X_low > tolerance:
        X = (X_low + X_high) / 2
        totalInjection = calculate_total_injection(X)
        
        if totalInjection > q_ref:
            X_high = X
        else:
            X_low = X

    # Use the found X to calculate final values
    X = X_low
    alternating_condition = (timeSteps % (dt1 + dt2)) < dt1
    time_limit_condition = timeSteps < X
    condition = alternating_condition & time_limit_condition
    injectionCriteria = condition.astype(int)
    totalInjection = np.sum(q * sat_per_second * injectionCriteria * dt)

    return X, injectionCriteria, totalInjection


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

    return row, col, data
        


def saturation(Q, P, dV, startEle, stopEle, convRate):
    nNodes = len(P)
    nUnk = 1 * nNodes 
    
    row = []
    col = []
    data = []
    
    b = np.zeros(nUnk, dtype = float)
    
    for ele in range(startEle, stopEle):
        n1, n2 = np.array(dataBase.iloc[ele, 1:3].values, dtype = int)
        q = np.abs(Q[ele])
        vol = dV[ele]
        
        # print(n1, n2)
        if P[n1] < P[n2]:
            n = n1
            m = n2
        elif P[n2] < P[n1]:
            n = n2
            m = n1
        
        # print(n, m)
        if n not in inlets and n not in contrastInlets:
            row.extend([n, n, n])
            col.extend([m, n, n])
            data.extend([q/vol, -q/vol, -convRate / rho_bolus])
        
        
        if m in inlets or m in contrastInlets:
            row.extend([m, m])
            col.extend([m, m])
            data.extend([0, 0])
            # data.extend([-q/vol, -convRate / rho_bolus])
            
    
    A = sp.coo_matrix((data, (row,col)), shape  = (nUnk, nUnk))
    
    return A, b, row, col ,data


def writeVTK(df_base, df_pts, pt_radius = None, pressure_data=None, temperature_data=None, concentration_data=None, damage_data=None, viscosity_data=None, chemData = None, *args, **kwargs):
    # Create VTK PolyData
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Add points to VTK PolyData
    for i in range(len(df_pts)):
        point = df_pts.iloc[i, [1, 2, 3]].tolist()
        points.InsertNextPoint(point)

    # Add lines to VTK PolyData
    for i in range(len(df_base)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, df_base.loc[i, 1])
        line.GetPointIds().SetId(1, df_base.loc[i, 2])
        lines.InsertNextCell(line)

    # Create a PolyData object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    # Add radius data if provided
    if pt_radius is not None:
        radii = vtk.vtkDoubleArray()
        radii.SetName("Radius")
        for i in range(len(df_pts)):
            radii.InsertNextValue(round(pt_radius[i],6))  
        polydata.GetPointData().AddArray(radii)
            

    # Add pressure data if provided
    if pressure_data is not None:
        prs = vtk.vtkDoubleArray()
        prs.SetName("Pressure")
        for i in range(len(df_pts)):
            prs.InsertNextValue(round(pressure_data[i],5))
        polydata.GetPointData().AddArray(prs)

    # Add temperature data if provided
    if temperature_data is not None:
        temp = vtk.vtkDoubleArray()
        temp.SetName("Temperature")
        for i in range(len(df_pts)):
            temp.InsertNextValue(temperature_data[i])
        polydata.GetPointData().AddArray(temp)

    # Add concentration data if provided
    if concentration_data is not None:
        conc = vtk.vtkDoubleArray()
        conc.SetName("Concentration")
        for i in range(len(df_pts)):
            conc.InsertNextValue(round(concentration_data[i],5))
        polydata.GetPointData().AddArray(conc)
        
    # Add damage data if provided
    if damage_data is not None:
        dmg = vtk.vtkDoubleArray()
        dmg.SetName("Damage")
        for i in range(len(df_pts)):
            dmg.InsertNextValue(round(damage_data[i],5))
        polydata.GetPointData().AddArray(dmg)
        
    # Add average viscosity if provided
    if viscosity_data is not None:
        visc = vtk.vtkDoubleArray()
        visc.SetName("averageViscosity")
        for i in range(len(df_pts)):
            visc.InsertNextValue(round(viscosity_data[i],5))
        polydata.GetPointData().AddArray(visc)
        
    # Add chemData if provided
    if chemData is not None:
        chem = vtk.vtkDoubleArray()
        chem.SetName("chemData")
        for i in range(len(df_pts)):
            chem.InsertNextValue(round(chemData[i],10))
        polydata.GetPointData().AddArray(chem)

    # Write PolyData to a VTK file
    filename = kwargs.get("filename", "output.vtk")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


if __name__=='__main__':
    
    
    # for iii in range(len(reference)):

    
    
    convRate = convRate_constant * timeConstant

    nNodes = np.max(dataBase.iloc[:,1:3].values) + 1
    nEle = len(dataBase)
    
    print('Number of elements = ', nEle)
    print('Number of nodes = ', nNodes)
    
    s_bolus = np.zeros(nNodes, dtype = float)
    s_blood = np.ones(nNodes, dtype = float)
    damage = np.zeros(nNodes, dtype = float)
    damage_convRate = np.zeros(nNodes, dtype = float)
    bolus_leaving = np.copy(damage_convRate)
    # index = 0
    
    # for tt, t in enumerate(timeSteps):
    myu_bolus_array = myu_bolus + myu_constant * s_bolus
    
    myu_mixed_node = s_blood * myu_blood + (s_bolus * myu_bolus_array)
    
    n1 = dataBase.iloc[:, 1].astype(int).values
    n2 = dataBase.iloc[:, 2].astype(int).values
    
    myu_mixed = (myu_mixed_node[n1] + myu_mixed_node[n2])/2
    
    kart = np.pi * dataBase.iloc[:, 4].values**4 / (8 * myu_mixed[:] * dataBase.iloc[:, 3].values)
                    
    row, col, data = flowSolver(kart, 0, nEle, gamma_a)
    
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
    
    
    outEle = dataBase[dataBase[1].isin(outlets) | dataBase[2].isin(outlets)]
    inEle = dataBase[dataBase[1].isin(inlets) | dataBase[2].isin(inlets)]
    outFlow = np.sum(np.abs(Qa[outEle.iloc[:, 0].values]))
    inFlow = np.sum(np.abs(Qa[inEle.iloc[:, 0].values]))
    errorFlow = inFlow - outFlow
    
    print('\nFlow Residual Error = ', "{:.1E}".format(np.linalg.norm(errorFlow)))
    print('mass conservation error     = ', "{:.1E}".format(errorFlow))
    
    print('Flow going in = ', round(inFlow * 6e7,3), 'ml per min' )
    print('Flow going out = ', round(outFlow * 6e7,3), 'ml per min')# , ' ml per min')
    
    injEle = dataBase[dataBase[1].isin(contrastInlets) | dataBase[2].isin(contrastInlets)]
    q_inj = np.average(np.abs(Qa[injEle.index.values]))
    
    print(q_inj*1E9)
    
    dVol_s = []
    
    
    for ele in range(0, nEle):
        n1, n2 = np.array(dataBase.iloc[ele, 1:3].values, dtype = int)
        q = np.abs(Qa[ele])
        vol = dV[ele]
        
        # print(n1, n2)
        if P_art[n1] < P_art[n2]:
            n = n1
            m = n2
        elif P_art[n2] < P_art[n1]:
            n = n2
            m = n1
        
        dVol_s.append(n)
        # s_node_vol_correlation.append([])
        
    dVol_s = np.array(dVol_s)
    injectVolLoc = np.argwhere(dVol_s == contrastInlets[0])
    # outletVolLoc = np.argwhere(dVol_s == outlets[:])
            
    
    avgQ_nodes = np.zeros(nNodes)
    for i in range(len(dataBase)):
        n1, n2 = dataBase.iloc[i,1:3]
        q = Qa[i]
        avgQ_nodes[int(n1)] = (avgQ_nodes[int(n1)] + abs(q))/2
        avgQ_nodes[int(n2)] = (avgQ_nodes[int(n2)] + abs(q))/2
        
    
    maxTime = 10 * 60
    timeSteps = np.arange(0, maxTime, dt)
    injectionCriteria = np.zeros(len(timeSteps))
    sat_per_second = 1# q_inj * dt
    timeRequired = injectedBolus / (sat_per_second * q_inj)
    number_dt = (timeRequired / dt )
    injectionCriteria[:int(number_dt)] = 1
    
    if timeRequired < dt:
        print("Need smaller timeSteps")
    
    simulationInjectedAmount = np.sum(injectionCriteria) * dt * sat_per_second * q_inj
    
    C, d, row, col, data = saturation(Qa, P_art, dV, 0, nEle, convRate)
    
    I = np.ones(len(P_art), dtype = float)
    I = sp.diags([I], offsets= [0])
    
    Ct = I - dt * C
    
    injectedSaturation = 0
    totalBolusInjected = 0
    bolusEscaping = np.zeros(len(outlets))
    bolusDepositing = np.zeros(nNodes)
    
    for tt, t in enumerate(timeSteps):
        s_bolus[contrastInlets] = injectionCriteria[tt] 
        d = s_bolus
        injectedSaturation = injectionCriteria[tt] * dt * q_inj
       
        x = spla.spsolve(Ct.tocsc(), d)
        
        thEmboError = Ct.dot(x) - d
        
        s_bolus = x[:nNodes]
        s_blood = 1 - s_bolus
        
        # MAP s_bolus to dV
        n1 = dataBase.iloc[:, 1].astype(int).values
        n2 = dataBase.iloc[:, 2].astype(int).values
        
        s_vol = (s_bolus[n1] + s_bolus[n2])/2.0
        
        bolus_leaving[outlets] = bolus_leaving[outlets] + s_bolus[outlets]
        damage_convRate = damage_convRate + s_bolus * convRate / rho_bolus * dt #* avgQ_nodes
        damage_convRate[contrastInlets] = 0
        
        totalBolusInjected = totalBolusInjected + injectedSaturation
        
        bolusDepositing = bolusDepositing + dt * s_bolus * convRate/rho_bolus
        
        bolusEscaping = bolusEscaping + s_bolus[outlets]
        
        if round(np.max(s_bolus),3) == 0:
            print('No More Bolus Left in the system')
            print('time = ', t)
            break
    
    
    outEle1 = dataBase[dataBase[1].isin(outlets)]
    outEle2 = dataBase[dataBase[2].isin(outlets)]
    
    n1 = outEle1.iloc[:,1].astype(int).values
    n2 = outEle2.iloc[:,2].astype(int).values
    
    bolusESX = 0
    for ii in range(len(outEle1)):
        index = outEle.iloc[ii,0]
        n = outEle1.iloc[ii,1]
        bolusESX = bolusESX + bolusEscaping[np.argwhere(outlets == n)[0][0]] * dt * dV[index]  # np.abs(Qa[index]) * dt
    
    for ii in range(len(outEle2)):
        index = outEle.iloc[ii,0]
        n = outEle2.iloc[ii,2]
        bolusESX = bolusESX + bolusEscaping[np.argwhere(outlets == n)[0][0]] * dt * dV[index]  #* np.abs(Qa[index]) * dt
    
    print('\n\nBolus Escaping = ', bolusESX * 1E9, ' uL', '\nDCACl escaping = ', epsilon * bolusESX * 1E9, ' uL')
    
    print('Total Deposited Bolus = ', np.sum(bolusDepositing))
    
    print('Total injected Bolus = ', q_inj * np.sum(injectionCriteria) * dt)
    
    
    depositedBolus = avgQ_nodes * dt * s_bolus
    
    
    s_fluid = 0
    s_deposited = 0
    s_depsoitedNodes = np.copy(damage_convRate)
    for ii in range(len(dVol_s)):
        if dVol_s[ii] not in injectionCriteria:
            s_fluid = s_fluid + s_bolus[dVol_s[ii]] * dV[ii]
            s_deposited = s_deposited + damage_convRate[dVol_s[ii]] * dV[ii]
    
    print(s_fluid/(q_inj * np.sum(injectionCriteria) * dt) * 100)
    print(s_deposited/(q_inj * np.sum(injectionCriteria) * dt) * 100)
    
        
    depositedPercentage = np.zeros(len(damage_convRate))
    
    for ii in range(len(dVol_s)):
        vol = dV[ii]
        sn = dVol_s[ii]
        depositedPercentage[sn] = damage_convRate[sn] * dV[ii]/(q_inj * np.sum(injectionCriteria) * dt)*100
    
    
    percentageBolusEscaping = bolusESX / (q_inj * np.sum(injectionCriteria) * dt)*100
    
    damage95 = (depositedPercentage > damageThreshold).astype(int)
    
    
    
    
    vtkFilename = steady + '_q_' + str(round(inFlow * 6e7,2)) + '_dmgTh_' + str(damageThreshold) + '_tC_' + str(timeConstant)+'_'
    writeVTK(dataBase, pts, pressure_data= P_art, 
              concentration_data=s_bolus, 
              damage_data = damage95, 
              viscosity_data = damage,
              chemData = depositedPercentage,
              filename =  folder + vtkFilename + '.vtk')
    
    
    
    # X = refDamage
    # Y = damage95
    
    # cm = confusion_matrix(X, Y)
    
    # TP = cm[1,1]
    # FP = cm[0,1]
    # FN = cm[1,0]
    # TN = cm[0,0]
    
    # precision = TP / (FP + TP) if (FP + TP) > 0 else 0
    # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # print('balanced accuracy = ', (recall + specificity)/2.0)
    
    # # Custom scoring function
    # score = (
    #     1 * max(0, percentageBolusEscaping - 0.01) +  # Penalize only if above threshold
    #     0 * (1 - precision) +  # Penalize false positives
    #     1 * (1 - recall)    # Penalize false negatives
    #     )
    
    # print('\npercentage of bolus escaping = ', round(bolusESX / (q_inj * np.sum(injectionCriteria) * dt)*100,3), ' %')
    
    # print('\nscore = ', score)
    
    # print('\ntime Constant = ', timeConstant,
    #       '\n cm = ', cm)
    
    
    # print('precision percentage = ', precision * 100)
    # print('recall percentage = ', recall * 100)
    
    # print('Max bolus deposited = ', np.max(depositedPercentage))
