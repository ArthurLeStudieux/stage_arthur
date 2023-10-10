import pandas as pd
import numpy as np
import math 
import os 
from datetime import datetime, timedelta
import pymap3d # python3 -m pip install pymap3d
# import random
from random import uniform, random
import pickle
from scipy.special import lambertw
from scipy import stats
from IPython.display import clear_output
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.plotting import figure, show, output_notebook, output_file, reset_output, save
from bokeh.layouts import gridplot, row, column
from bokeh.models import Range1d
from casadi import *
from pylab import * 



def read_data(parameters):
    """Returns the dataframe corresponding to the given parameters (one user, one trajectory)

    Args:
        parameters (dictionnary): _description_

    Raises:
        NameError: _description_
    """
    if parameters['dataset'] == 'cabspotting':
        local_url="../datasets/cabspotting/" + parameters['dataset'] + "-tree/" + parameters['user'] + ".csv"
        df = pd.read_csv(local_url,names=["latitude","longitude","timestamp"])
    elif parameters['dataset'] == 'geolife':
        data_path = '../datasets/Geolife/Data/'
        file = os.listdir(data_path)[parameters['user']]
        traj = os.listdir(data_path + str(file) + '/Trajectory/')[parameters['trajectory_index']]
        filepath = os.path.join(data_path + str(file) + '/Trajectory/', traj)
        with open(filepath, 'r') as f:
            lines = f.readlines()[6:] #on s'en fiche des 6 premières lignes, (spécifique à Geolife)
        data = np.zeros((len(lines),5))
        min_time = float(lines[0].split(',')[4])
        i=0
        for line in lines:
            line = line.split(',')
            latitude = float(line[0])
            longitude = float(line[1])
            altitude = float(line[3])
            elapsedtime = (float(line[4]) - min_time)*(3600*24)
            data[i] = ([parameters['user'], latitude, longitude, altitude, elapsedtime])
            i+= 1
        df = pd.DataFrame(data, columns=['user','latitude', 'longitude', 'altitude', 'elapsedtime'])

    else:
        raise NameError('the dataset parameter does not correspond to any of the available datasets')
    return(df)





def process_data(parameters, data):
    """Returns the same dataset along with geodetic's x,y coordinates 

    Args:
        parameters (_type_): _description_
        data (_type_): _description_
    """
    ell_grs80 = pymap3d.Ellipsoid('grs80') 
    data['altitude']=np.zeros(len(data.index))
    lat0, lon0, h0 = data['latitude'][0],data['longitude'][0],data['altitude'][0]
    if parameters['dataset'] == 'cabspotting':
        #data['time']=[datetime.fromtimestamp(tstamp/1000) for tstamp in data['timestamp']]
        data['elapsedtime']=(data['timestamp']-data['timestamp'][0])/1000
        data['x'], data['y'], data['u_enu'] = pymap3d.geodetic2enu(data['latitude'], data['longitude'], data['altitude'], lat0, lon0, h0, ell=ell_grs80)
    else: #Geolife's altitude is expressed in feet
        feet_meter_ratio = 0.3048
        data['x'], data['y'], data['u_enu'] = pymap3d.geodetic2enu(data['latitude'], data['longitude'], data['altitude']*feet_meter_ratio, lat0, lon0, h0, ell=ell_grs80)
    return(data)



# from casadi import *
# from pylab import * 
def data_save(data, parameters):
    """ Restricts the considered data to a fixed interval of time [Tmin, Tmax] (chosen in the parameters),
    then saves the new data in a pickle file. Introduces the tools necessary for the p mpc-H algorithm

    Args:
        data (dataframe): _description_
        parameters (dictionnary): _description_
    """
    Tmin, Tmax = parameters["Tmin"], parameters["Tmax"]
    nbuf = int(parameters['POI_duration']* 60 / parameters['sampling_time'])
    xr = data['x'][(data['elapsedtime'] >= Tmin) & (data['elapsedtime'] < Tmax)]
    yr = data['y'][(data['elapsedtime'] >= Tmin) & (data['elapsedtime'] < Tmax)]
    ur = data['u_enu'][(data['elapsedtime'] >= Tmin) & (data['elapsedtime'] < Tmax)]
    Timer = data['elapsedtime'][(data['elapsedtime'] >= Tmin) & (data['elapsedtime'] < Tmax)]
    data_timed = data[(data['elapsedtime'] >= Tmin) & (data['elapsedtime'] < Tmax)]
    ts = parameters['sampling_time']  # Sampling time
    tmax = Timer.iloc[-1]  # Final time 
    nt = int(tmax/ts)+1

    newx = []
    newy = []
    newu = []
    newb = []
    newTimer = []
    for i in range(0,nt):
        xt = data['x'][(data['elapsedtime'] >= i*ts) & (data['elapsedtime'] < (i+1)*ts)].mean() 
        yt = data['y'][(data['elapsedtime'] >= i*ts) & (data['elapsedtime'] < (i+1)*ts)].mean() 
        ut = data['u_enu'][(data['elapsedtime'] >= i*ts) & (data['elapsedtime'] < (i+1)*ts)].mean() 
        if np.isnan(xt) == False:
            newx += [xt]
            newy += [yt]
            newu += [ut]
            newb += [1]
        else:  # Fill with zeros when no transmitted
            newx += [0]
            newy += [0]
            newu += [0]
            newb += [0]
        newTimer += [(i+1)*ts]

    xs = np.array(newx)
    ys = np.array(newy)
    bs = np.array(newb)
    us = np.array(newu)
    Time = np.array(newTimer)

    save_data_raw = pd.DataFrame()
    save_data_raw['Timer'] = Timer
    save_data_raw['xr'] = xr
    save_data_raw['yr'] = yr
    sol_file = parameters['output_dir']
    pickle.dump( save_data_raw, open( sol_file+"/save_data_raw.pkl", "wb" ) )

    save_data = pd.DataFrame()
    save_data['Time'] = Time
    save_data['xs'] = xs
    save_data['ys'] = ys
    save_data['us'] = us
    save_data['bs'] = bs
    pickle.dump( save_data, open( sol_file+"/save_data.pkl", "wb" ) )
    save_data_raw = pickle.load( open( sol_file+"/save_data_raw.pkl", "rb" ) )
    save_data = pickle.load( open( sol_file+"/save_data.pkl", "rb" ) )

    x_obf=np.zeros_like(xs)
    y_obf=np.zeros_like(ys)
    for i in range(nt):
        x_obf[i] ,y_obf[i], r =GeoI_LPPM(xs[i],ys[i],0.005)
        
    delta=np.zeros((2,nt))
    delta[0,]=x_obf-xs
    delta[1,]=y_obf-ys

    X = np.array([0]*nbuf)
    Y = np.array([0]*nbuf)
    N = np.array([0]*nbuf)

    system = pd.DataFrame()
    system['X'] = X
    system['Y'] = Y
    system['N'] = N

    system = reset_system(system)
    barxk = []
    baryk = []
    privk = []
    system_obf = reset_system(system)
    barxk_obf = []
    baryk_obf = []
    privk_obf = []

    util = []
    for i in range(0,nt,1):
        bi = bs[i] 
        # Update the real system
        system = system_update(system,xs[i],ys[i],bi)
        barx, bary, priv = privacy(system)
        barxk += [barx] 
        baryk += [bary] 
        privk += [priv] 
        # Update the obfuscated system
        system_obf = system_update(system_obf,x_obf[i],y_obf[i],bi)
        barx_obf, bary_obf, priv_obf = privacy(system_obf)
        barxk_obf += [barx_obf] 
        baryk_obf += [bary_obf] 
        privk_obf += [priv_obf] 
        # Compute utility
        util += [utility(delta[0,i],delta[1,i])*bi]

    Data_real = pd.DataFrame()
    Data_real['x'] = xs
    Data_real['y'] = ys
    Data_real['b'] = bs
    Data_real['barx'] = barxk
    Data_real['bary'] = baryk
    Data_real['priv'] = privk
    Data_real['u_enu'] = us

    Data_obf = pd.DataFrame()
    Data_obf['x'] = x_obf
    Data_obf['y'] = y_obf
    Data_obf['U'] = delta[0,]
    Data_obf['V'] = delta[1,]
    Data_obf['barx'] = barxk_obf
    Data_obf['bary'] = baryk_obf
    Data_obf['priv'] = privk_obf
    Data_obf['util'] = util
    Data_obf['u_enu'] = us

    pickle.dump( Data_real, open( sol_file+"/Data_real.pkl", "wb" ) )
    pickle.dump( Data_obf, open( sol_file+"/Data_obf.pkl", "wb" ) )
    # Control
    u = MX.sym("u")
    v = MX.sym("v")
    b = MX.sym("b")

    # State
    x = MX.sym("x",nbuf)
    y = MX.sym("y",nbuf)
    nx = MX.sym("nx",nbuf)

    # Matrices
    A = np.zeros((nbuf, nbuf))
    B = np.zeros((nbuf, 1))
    for i in range(nbuf-1):
        A[i,i+1] = 1

    B[nbuf-1,0] = 1
    A.tolist()
    B.tolist()
    A = DM(A)
    B = DM(B)

    # Dynamics
    xplus = A@x + B@u
    yplus = A@y + B@v
    nplus = A@nx + B@b

    # Discrete time dynamics function
    F = Function('F', [x,u],[xplus])

    #Privacity function
    O = DM(np.ones((1,nbuf)))
    xbar = (O@(x*nx))/((O@nx)+1e-8)
    ybar = O@(y*nx)/((O@nx)+1e-8)
    P = (O@( sqrt((x-xbar)**2 + (y-ybar)**2)*nx) )/((O@nx)+1e-8)
    J = Function('J', [x,y,nx],[xbar, ybar, P],['x','y','nx'],['xbar', 'ybar','P'])
    return(xs, ys, bs, us, J, F, ts, util)







##########################################              DYNAMIC SYSTEM             ##########################################
#This whole part directly comes from the p mpc-H algorithm in the notebook
def system_update(system,x,y,b):
    dict = pd.DataFrame({'X': [x], 'Y': [y], 'N': [b]}) #sans pd.dataframe, sans les listes
    system = pd.concat([system, dict], ignore_index = True, axis = 0).drop(index=[0]).reset_index(drop=True) #system.concat(dict,ignore_index,pasd'axis)
    return system

def privacy(system):
    n = system['N'].sum()
    if n == 0:
        barx = 0
        bary = 0
        priv = 0
    else:
        barx = np.dot(system['X'], system['N'])/n
        bary = np.dot(system['Y'], system['N'])/n
        #priv = np.sqrt(np.dot((system['X'] - barx)**2 + (system['Y'] - bary)**2, system['N'])/n) 
        priv = np.dot(np.sqrt((system['X'] - barx)**2 + (system['Y'] - bary)**2), system['N'])/n 
    return barx, bary, priv

def utility(x,y):
    return np.sqrt(x**2 + y**2)

def reset_system(system):
    n = len(system)
    X = np.array([0]*n)
    Y = np.array([0]*n)
    N = np.array([0]*n)

    system['X'] = X
    system['Y'] = Y
    system['N'] = N
    return system



###############################################              GEO-I             ###############################################

def GeoI_LPPM(x,y,epsilon):
    theta_GeoI = uniform(0,2*math.pi)
    r_GeoI = -1/epsilon*(np.real(lambertw((random()-1)/math.exp(1),k=-1))+1)
    x_ctrl = x + r_GeoI*math.cos(theta_GeoI)
    y_ctrl = y + r_GeoI*math.sin(theta_GeoI)
    return x_ctrl, y_ctrl, r_GeoI 



###############################################              FLAIR             ###############################################
def FLAIR_insert_1d(x0, x1, x, t0, t1, t, A0, A_min, A_max, S, A, T, epsilon):
    """Inserts a new sample (x,t) into the previous model if possible. Creates a new model otherwise.

    Args:
        x0 (_type_): first sample of the last model
        x1 (_type_): last sample of the last model
        x (_type_): sample to be inserted
        t0 (_type_): first time..
        t1 (_type_): _description_
        t (_type_): _description_
        A0 (_type_): linear coef of the current model
        A_min (_type_): _description_
        A_max (_type_): _description_
        S (_type_): stores x
        A (_type_): stores linear coefs
        T (_type_): stores time

    Returns:
        _type_: _description_
    """
    x_delta, t_delta = x -x0, t - t0
    
    A_t = x_delta/(t_delta+1e-8)
    if A_min <= A_t and A_t <= A_max:
        
        A0 = A_t
        A_min = max(A_min, (x_delta-epsilon)/(t_delta + 1e-12))
        A_max = min(A_max, (x_delta+epsilon)/(t_delta + 1e-12))
    else: 
        S.append(x0)
        A.append(A0)
        T.append(t0)
        t0 = t
        x0, t0 = x1, t1
        x_delta, t_delta = x -x0, t -t0
        A0 = x_delta/(t_delta+1e-12)
        A_min = (x_delta-epsilon)/(t_delta + 1e-12)
        A_max = (x_delta+epsilon)/(t_delta + 1e-12)
    x1, t1 = x, t 
    return (x0, t0, x1, t1, A0, A_min, A_max)



def FLAIR_PMRD(data, mode, epsilon):
    """transforms the mobility data of the different users just as in the FLAIR paper

    Args:
        data (dataframe): the data
        mode(str): either 'x', 'y', or "u_enu"
        epsilon(float): FLAIR's epsilon
    """
    
    X = [] #will stock the points of the models
    Ax = [] #will stock the linear coefficients
    Tx = [] #will stock the time at which a new model begins
    t_min = data['timestamp'][0]
    
    if data[mode].shape[0] >= 2: #we never really have to make this distinction
            x0 = data[mode][0]
            x1 = data[mode][1]
            t0 = (data['timestamp'][0] - t_min) / 1000 #everything in seconds (i realized afterwards elapsedtime already did that)
            t1 = (data['timestamp'][1] - t_min) /1000
            A0_x = (x1-x0)/(t1-t0 +1e-12) #On some datasets we could have the exact same time on two different samples
            A_min_x = (x1-epsilon-x0)/(t1-t0 +1e-12)
            A_max_x = (x1+epsilon-x0)/(t1-t0 +1e-12)
            for i in range (2,data.shape[0]):
                x = data[mode][i]
                t = (data['timestamp'][i] - t_min)/ 1000
                x0, t0, x1, t1, A0_x, A_min_x, A_max_x = FLAIR_insert_1d(x0, x1, x, t0, t1, t, A0_x, A_min_x, A_max_x, X, Ax, Tx, epsilon)
            X.append(x0)
            Ax.append(A0_x)
            Tx.append(t0)
    return(X,Ax,Tx)



def prediction(horizon, coef, last_point, time_gap):
    """Returns an array of size horizon for which index i has value = last_point + (time_gap + 30*i)*(coef)

    Args:
        last_point (float): the last transmitted position (either x-wise or y -wise)
        horizon (int)
        coef (float): the linear coefficient of the current model 

    Returns:
        Xf: the array of predicted positions
    """
    #print(coef, time_gap, last_point)
    return(np.array([last_point + (time_gap + 30*i)*(coef) for i in range(horizon)]))

def FLAIR_insert(x0, x1, x, t0, t1, t, A0, A_min, A_max, epsilon):
    """Inserts a new sample (x,t) into the previous model if possible. Creates a new model otherwise.

    Args:
        x0 (_type_): first sample of the last model
        x1 (_type_): last sample of the last model
        x (_type_): sample to be inserted
        t0 (_type_): first time..
        t1 (_type_): _description_
        t (_type_): _description_
        A0 (_type_): linear coef of the current model
        A_min (_type_): _description_
        A_max (_type_): _description_
        S (_type_): stores x
        A (_type_): stores linear coefs
        T (_type_): stores time

    Returns:
        _type_: _description_
    """
    x_delta, t_delta = x -x0, t - t0
    
    A_t = x_delta/(t_delta+1e-8)
    if A_min <= A_t and A_t <= A_max:
        
        A0 = A_t
        A_min = max(A_min, (x_delta-epsilon)/(t_delta + 1e-12))
        A_max = min(A_max, (x_delta+epsilon)/(t_delta + 1e-12))
    else: 
        t0 = t
        x0, t0 = x1, t1
        x_delta, t_delta = x -x0, t -t0
        A0 = x_delta/(t_delta+1e-12)
        A_min = (x_delta-epsilon)/(t_delta + 1e-12)
        A_max = (x_delta+epsilon)/(t_delta + 1e-12)
    x1, t1 = x, t 
    return (x0, t0, x1, t1, A0, A_min, A_max)





def predict_xf(current_FLAIR_model, tk, xs, horizon, model_defined, epsilon, bs, ts):
    """ predicts the next samples, using flair as a predictor, using an harmonized trajectory (with a value each 30s)

    Args:
        current_FLAIR_model (_type_): _description_
        tk (_type_): _description_
        xs (_type_): _description_
        horizon (_type_): _description_
        model_defined (_type_): _description_
        epsilon (_type_): _description_
    """
    
    if tk == 0: #For tk = 0 , we don't have any flair model yet, we'll consider Ax = Ay = 0
        x0 = xs[tk]
        t0 = tk
        

        A0_x = 0
        return(prediction(horizon, A0_x, x0, tk), model_defined, (A0_x, t0, x0))
    elif (not model_defined):
        if bs[tk] == 0:
            return(prediction(horizon, current_FLAIR_model[0], current_FLAIR_model[2], ts* tk - current_FLAIR_model[1]), model_defined, current_FLAIR_model)
        else:
            x0, t0 = current_FLAIR_model[2],current_FLAIR_model[1]
            model_defined = True
            x1 = xs[tk]
            t1 = ts*tk
            A0_x = (x1-x0)/(t1-t0 +1e-12)
            A_min_x = (x1-epsilon-x0)/(t1-t0 +1e-12)
            A_max_x = (x1+epsilon-x0)/(t1-t0 +1e-12)
            return(prediction(horizon, A0_x, current_FLAIR_model[2], ts*tk - current_FLAIR_model[1]), model_defined, (A0_x, t0, x0, x1, t1, A_min_x, A_max_x))
    else: #We have a FLAIR model, we now want to modify it
        if bs[tk] == 0: #No new sample, we continue with the previous model
            return(prediction(horizon, current_FLAIR_model[0], current_FLAIR_model[2], ts*tk - current_FLAIR_model[1]), model_defined, current_FLAIR_model)
        else:
            x0, t0 = current_FLAIR_model[2],current_FLAIR_model[1]
            x1, t1 = current_FLAIR_model[3],current_FLAIR_model[4]
            print(t0, t1)
            A0_x, A_min_x, A_max_x = current_FLAIR_model[0], current_FLAIR_model[5], current_FLAIR_model[6]
            x0, t0, x1, t1, A0_x, A_min_x, A_max_x = FLAIR_insert(x0, x1, xs[tk], t0, t1, ts*tk, A0_x, A_min_x, A_max_x, epsilon)
            #print('\n coef directeur: ')
            return(prediction(horizon, A0_x, x0, ts*tk - t0), model_defined, (A0_x, t0, x0, x1, t1, A_min_x, A_max_x)) 
        


def bs_pred(H, bs):
    """Returns bs except the last H samples are always equal to 1 (we consider that the signal is always transmitted in the FLAIR's case
    since FLAIR's prediction is just a line)
    Args:
        H (int): the horizon
    """
    bs_copy = bs.copy()
    for i in range (1,H+1):
        bs_copy[-i] = 1
    return(bs_copy)




def FLAIR_eval(horizon, x, y):
    mean_loss_x = 0
    mean_loss_y = 0
    for k in range (x.shape[0]):
        predict_xf((0,0,0), k, )


###############################################            SOLVE MPC           ###############################################

def solve_mpc(xs,ys,bs,us,horizon,nbuf,util_bound, J, F):
    """Solves the optimisation problem such as stated in the p mpc-H paper.

    Args:
        xs (_type_): _description_
        ys (_type_): _description_
        bs (_type_): _description_
        us (_type_): _description_
        horizon (_type_): _description_
        nbuf (_type_): _description_
        util_bound (_type_): _description_
        J (_type_): _description_
        F (_type_): _description_

    Returns:
        _type_: _description_
    """
    nt = len(xs) #c'est plus le même nt là du coup
    x_mpc = []
    y_mpc = []
    U_mpc = []
    V_mpc = []
    barx_mpc = []
    bary_mpc = []
    priv_mpc = []
    util_mpc = []
    time_mpc = []
    X0 = np.array([0]*nbuf).tolist()
    Y0 = np.array([0]*nbuf).tolist()
    N0 = np.array([0]*nbuf).tolist()

    system_opt = pd.DataFrame()
    system_opt['X'] = X0
    system_opt['Y'] = Y0
    system_opt['N'] = N0

    for tk in range(0,nt-horizon):#nt-horizon):

        if bs[tk] == 0:
            U_mpc += [0]
            V_mpc += [0]
            x_mpc += [xs[tk]]
            y_mpc += [ys[tk]]
            system_opt = system_update(system_opt, xs[tk], ys[tk], 0)
            barxt, baryt, privt = privacy(system_opt)
            barx_mpc += [barxt] 
            bary_mpc += [baryt] 
            priv_mpc += [privt] 
            # Compute utility
            util_mpc += [0]
            time_mpc += [0]
            #system_opt = system_update(system_opt, xs[tk], ys[tk], 0)

        else:
            time_0= time.time()
            X0 = system_opt['X'].tolist()
            Y0 = system_opt['Y'].tolist()
            N0 = system_opt['N'].tolist()

            # Initial conditions
            Jk = 0
            w=[]
            w0 = []
            lbw = []
            ubw = []
            g=[]
            lbg = []
            ubg = []

            # Constraints on input
            U = MX.sym("U",1)
            V = MX.sym("V",1)
            w += [U, V]
            lbw += [-inf]*2
            ubw += [inf]*2
            w0 += [0]*2
            g   += [U**2 + V**2]
            lbg += [0]
            ubg += [util_bound[tk]**2]

            # Initial conditions
            Xk = MX.sym("Xk",nbuf)
            w += [Xk]
            lbw += X0
            ubw += X0
            w0 += X0
            Yk = MX.sym("Yk",nbuf)
            w += [Yk]
            lbw += Y0
            ubw += Y0
            w0 += Y0
            Nk = MX.sym("Nk",nbuf)
            w += [Nk]
            lbw += N0
            ubw += N0
            w0 += N0

            # Future values of x, y and n
            Xf = xs[tk:tk+horizon]
            Xf = Xf[::].tolist()
            Xkfut = MX.sym("Xkfut",horizon) 
            w += [Xkfut]
            lbw += Xf
            ubw += Xf
            w0 += Xf
            Yf = ys[tk:tk+horizon]
            Yf = Yf[::].tolist()
            Ykfut = MX.sym("Ykfut",horizon) 
            w += [Ykfut]
            lbw += Yf
            ubw += Yf
            w0 += Yf
            Nf = bs[tk:tk+horizon]
            Nf = Nf[::].tolist()
            Nkfut = MX.sym("Nkfut",horizon) 
            w += [Nkfut]
            lbw += Nf
            ubw += Nf
            w0 += Nf

            # Apply the first input 
            Xk = F(Xk,Xkfut[0]+U)
            Yk = F(Yk,Ykfut[0]+V)
            Nk = F(Nk,1)
            Jt = J(x=Xk,y=Yk,nx=Nk)
            Jk += Jt['P']
            for k in range(horizon-1):
                # New NLP variable for the control
                Uk = MX.sym('U_' + str(k))
                Vk = MX.sym('V_' + str(k))
                w   += [Uk, Vk]
                lbw += [-util_bound[tk+k+1]*Nf[k+1]]*2
                ubw += [util_bound[tk+k+1]*Nf[k+1]]*2
                w0 += [0]*2

                g   += [Uk**2 + Vk**2]
                lbg += [0]
                ubg += [util_bound[tk+k+1]**2]

                Xk = F(Xk,Xkfut[k+1]+Uk)
                Yk = F(Yk,Ykfut[k+1]+Vk)
                Nk = F(Nk,Nkfut[k+1])
                Jt = J(x=Xk,y=Yk,nx=Nk)
                Jk += Jt['P']


            prob = {'f': -Jk, 'x': vertcat(*w), 'g': vertcat(*g)}
            solver = nlpsol('solver', 'ipopt', prob)
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            w_opt = sol['x'].full().flatten()
            J_opt = -sol['f']

            # Plot the solution
            u_opt = w_opt[0]
            v_opt = w_opt[1]
            U_mpc += [u_opt]
            V_mpc += [v_opt]
            x_mpc += [Xf[0] + u_opt]
            y_mpc += [Yf[0] + v_opt]
            system_opt = system_update(system_opt, Xf[0] + u_opt, Yf[0] + v_opt, 1)
            barxt, baryt, privt = privacy(system_opt)
            barx_mpc += [barxt] 
            bary_mpc += [baryt] 
            priv_mpc += [privt] 
            # Compute utility
            util_mpc += [utility(u_opt,v_opt)]
            time_mpc += [time.time()-time_0]
            #system_opt = system_update(system_opt, Xf[0] + u_opt, Yf[0] + v_opt, 1)
   
    clear_output(wait=False)

    data_mpc = pd.DataFrame()
    data_mpc['x'] = x_mpc
    data_mpc['y'] = y_mpc
    data_mpc['u_enu']= us[:len(x_mpc)]
    data_mpc['U'] = U_mpc
    data_mpc['V'] = V_mpc
    data_mpc['barx'] = barx_mpc
    data_mpc['bary'] = bary_mpc
    data_mpc['priv'] = priv_mpc
    data_mpc['util'] = util_mpc
    data_mpc['time'] = time_mpc
    
    return data_mpc#U_mpc, V_mpc, x_mpc, y_mpc, barx_mpc, bary_mpc, priv_mpc, util_mpc 



###############################################         SOLVE MPC FLAIR          ###############################################

def solve_mpc_FLAIR(xs,ys,bs,us,ts,horizon,nbuf,util_bound, epsilon, J, F):
    """Solves the optimisation problem, except the data of the future positions is replaced by FLAIR's predictions

    Args:
        xs (_type_): _description_
        ys (_type_): _description_
        bs (_type_): _description_
        us (_type_): _description_
        ts (_type_): _description_
        horizon (_type_): _description_
        nbuf (_type_): _description_
        util_bound (_type_): _description_
        epsilon (_type_): _description_
        J (_type_): _description_
        F (_type_): _description_

    Returns:
        _type_: _description_
    """
    current_FLAIR_model_x, current_FLAIR_model_y = (0,0,0), (0,0,0)
    model_defined_x = False
    model_defined_y = False
    model = 0
    Xf_FLAIR, Tx_FLAIR, X_FLAIR, Ax_FLAIR = [], [], [], [] #Juste là pour vérifier que tout se passe bien
    nt = len(xs) #c'est plus forcément le même nt qu'avant là du coup
    x_mpc = []
    y_mpc = []
    U_mpc = []
    V_mpc = []
    barx_mpc = []
    bary_mpc = []
    priv_mpc = []
    util_mpc = []
    time_mpc = []
    X0 = np.array([0]*nbuf).tolist()
    Y0 = np.array([0]*nbuf).tolist()
    N0 = np.array([0]*nbuf).tolist()

    system_opt = pd.DataFrame()
    system_opt['X'] = X0
    system_opt['Y'] = Y0
    system_opt['N'] = N0

    for tk in range(0,nt-horizon):#nt-horizon):

        if bs[tk] == 0:
            U_mpc += [0]
            V_mpc += [0]
            x_mpc += [xs[tk]]
            y_mpc += [ys[tk]]
            system_opt = system_update(system_opt, xs[tk], ys[tk], 0)
            barxt, baryt, privt = privacy(system_opt)
            barx_mpc += [barxt] 
            bary_mpc += [baryt] 
            priv_mpc += [privt] 
            # Compute utility
            util_mpc += [0]
            time_mpc += [0]
            #system_opt = system_update(system_opt, xs[tk], ys[tk], 0)

        else:
            time_0= time.time()
            X0 = system_opt['X'].tolist()
            Y0 = system_opt['Y'].tolist()
            N0 = system_opt['N'].tolist()

            # Initial conditions
            Jk = 0
            w=[]
            w0 = []
            lbw = []
            ubw = []
            g=[]
            lbg = []
            ubg = []

            # Constraints on input
            U = MX.sym("U",1)
            V = MX.sym("V",1)
            w += [U, V]
            lbw += [-inf]*2
            ubw += [inf]*2
            w0 += [0]*2
            g   += [U**2 + V**2]
            lbg += [0]
            ubg += [util_bound[tk]**2]

            # Initial conditions
            Xk = MX.sym("Xk",nbuf)
            w += [Xk]
            lbw += X0
            ubw += X0
            w0 += X0
            Yk = MX.sym("Yk",nbuf)
            w += [Yk]
            lbw += Y0
            ubw += Y0
            w0 += Y0
            Nk = MX.sym("Nk",nbuf)
            w += [Nk]
            lbw += N0
            ubw += N0
            w0 += N0

           
            Xf, model_defined_x, current_FLAIR_model_x = predict_xf(current_FLAIR_model_x, tk, xs, horizon, model_defined_x, epsilon, bs, ts)
            #current_FLAIR_model looks like: A0_x, t0, x0, x1, t1, A_min_x, A_max_x
            if tk!=0:
                Ax_FLAIR.append(current_FLAIR_model_x[0])
                Tx_FLAIR.append(tk*30)
                X_FLAIR.append(current_FLAIR_model_x[3])
            Xf_FLAIR.append(Xf)
            Xf = Xf[::].tolist()
            Xkfut = MX.sym("Xkfut",horizon) 
            w += [Xkfut]
            lbw += Xf
            ubw += Xf
            w0 += Xf

            #Yf = ys[tk:tk+horizon]
            Yf, model_defined_y, current_FLAIR_model_y = predict_xf(current_FLAIR_model_y, tk, ys, horizon, model_defined_y, epsilon, bs, ts)
            Yf = Yf[::].tolist()
            Ykfut = MX.sym("Ykfut",horizon) 
            w += [Ykfut]
            lbw += Yf
            ubw += Yf
            w0 += Yf
            # à changer
            Nf = np.concatenate((np.array([bs[tk]]), np.ones(horizon - 1))) #FLAIR's prediction being continuous, we'll consider
            #that every sample is transmitted
            #Nf = bs[tk:tk+horizon]
            Nf = Nf[::].tolist()
            Nkfut = MX.sym("Nkfut",horizon) 
            w += [Nkfut]
            lbw += Nf
            ubw += Nf
            w0 += Nf

            # Apply the first input 
            Xk = F(Xk,Xkfut[0]+U)
            Yk = F(Yk,Ykfut[0]+V)
            Nk = F(Nk,1)
            Jt = J(x=Xk,y=Yk,nx=Nk)
            Jk += Jt['P']
            for k in range(horizon-1):
                # New NLP variable for the control
                Uk = MX.sym('U_' + str(k))
                Vk = MX.sym('V_' + str(k))
                w   += [Uk, Vk]
                lbw += [-util_bound[tk+k+1]*Nf[k+1]]*2
                ubw += [util_bound[tk+k+1]*Nf[k+1]]*2
                w0 += [0]*2

                g   += [Uk**2 + Vk**2]
                lbg += [0]
                ubg += [util_bound[tk+k+1]**2]

                Xk = F(Xk,Xkfut[k+1]+Uk)
                Yk = F(Yk,Ykfut[k+1]+Vk)
                Nk = F(Nk,Nkfut[k+1])
                Jt = J(x=Xk,y=Yk,nx=Nk)
                Jk += Jt['P']


            prob = {'f': -Jk, 'x': vertcat(*w), 'g': vertcat(*g)}
            solver = nlpsol('solver', 'ipopt', prob)
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            w_opt = sol['x'].full().flatten()
            J_opt = -sol['f']

            # Plot the solution
            u_opt = w_opt[0]
            v_opt = w_opt[1]
            U_mpc += [u_opt]
            V_mpc += [v_opt]
            x_mpc += [Xf[0] + u_opt]
            y_mpc += [Yf[0] + v_opt]
            system_opt = system_update(system_opt, Xf[0] + u_opt, Yf[0] + v_opt, 1)
            barxt, baryt, privt = privacy(system_opt)
            barx_mpc += [barxt] 
            bary_mpc += [baryt] 
            priv_mpc += [privt] 
            # Compute utility
            util_mpc += [utility(u_opt,v_opt)]
            time_mpc += [time.time()-time_0]
            #system_opt = system_update(system_opt, Xf[0] + u_opt, Yf[0] + v_opt, 1)
   
    clear_output(wait=False)

    data_mpc = pd.DataFrame()
    data_mpc['x'] = x_mpc
    data_mpc['y'] = y_mpc
    data_mpc['u_enu']= us[:len(x_mpc)]
    data_mpc['U'] = U_mpc
    data_mpc['V'] = V_mpc
    data_mpc['barx'] = barx_mpc
    data_mpc['bary'] = bary_mpc
    data_mpc['priv'] = priv_mpc
    data_mpc['util'] = util_mpc
    data_mpc['time'] = time_mpc
    
    
    #return data_mpc, Ax_FLAIR, Tx_FLAIR, X_FLAIR, Xf_FLAIR
    return data_mpc






###############################################         FONCTIONS PLOT          ###############################################

def plot_traj(x_coords, y_coords, tk, file_name, file_format='png'):
    """Creates a png file plotting the trajectory

    Args:
        x_coords (list): Corresponds to xs (so contains the 0)
        y_coords (list): ys
        file_name (_type_): where we store the image
        file_format (str, optional): Defaults to 'png'.
    """
    T = np.array([tk*i for i in range(len(x_coords))])
    non_zero_indices = np.nonzero(np.logical_and(x_coords != 0, y_coords != 0))
    x_non_zero = x_coords[non_zero_indices]
    y_non_zero = y_coords[non_zero_indices]
    T = T[non_zero_indices]


    #Traj X,Y
    fig, ax = plt.subplots()
    ax.plot(x_non_zero, y_non_zero, 'bo') 
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_title('Trajectory')
    # Saving the plot as an image file
    file_path = os.path.join('images', f'{file_name}.{file_format}')
    plt.savefig(file_path, format=file_format)
    plt.close(fig)


    #Traj T,X 
    fig, ax = plt.subplots()
    ax.plot(T, x_non_zero, 'bo') 
    ax.set_xlabel('Time')
    ax.set_ylabel('X Coordinates')
    ax.set_title('X Coordinates')
    # Saving the plot as an image file
    file_path = os.path.join('images', f'{"xCoordinates"}.{file_format}')
    plt.savefig(file_path, format=file_format)
    plt.close(fig)

    #Traj T,Y
    fig, ax = plt.subplots()
    ax.plot(T, y_non_zero, 'bo') 
    ax.set_xlabel('Time')
    ax.set_ylabel('Y Coordinates')
    ax.set_title('Y coordinates')
    # Saving the plot as an image file
    file_path = os.path.join('images', f'{"yCoordinates"}.{file_format}')
    plt.savefig(file_path, format=file_format)
    plt.close(fig)






from bokeh.io import output_file, export
def plot_privacy(Data_real, Data_obf, Data_MPC, Data_MPC_FLAIR2, h):
    """Plots the privacy metrics for the initial trajectory, after Geo-I obfuscation, and after p mpc-H and mpc FLAIR obfuscations.
    The plots are saved 

    Args:
        Data_real (_type_): _description_
        Data_obf (_type_): _description_
        Data_MPC (_type_): _description_
        Data_MPC_FLAIR2 (_type_): _description_
        h (_type_): _description_
    """
    privk = Data_real['priv']
    privk_obf = Data_obf['priv']
    Time = np.array([30*i for i in range(privk.shape[0])])


    fig = bokeh.plotting.figure(width = 900, height = 450)
    fig.line(Time, privk, line_color='red', legend_label="p")
    fig.circle(Time, privk, color='red', legend_label="p")
    fig.line(Time, privk_obf, line_color='orange', legend_label="geo I")
    fig.circle(Time, privk_obf, color='orange', legend_label="geo I")
    priv_mpc_FLAIR = Data_MPC_FLAIR2.priv
    fig.line(Time[0:len(priv_mpc_FLAIR)], priv_mpc_FLAIR, line_color="darkviolet", legend_label="p mpc FLAIR"+str(h))
    fig.circle(Time[0:len(priv_mpc_FLAIR)], priv_mpc_FLAIR[0:len(priv_mpc_FLAIR)], color="darkviolet", legend_label="p mpc FLAIR"+str(h))
    priv_mpc = Data_MPC.priv
    fig.line(Time[0:len(priv_mpc)], priv_mpc, line_color="forestgreen", legend_label="p mpc "+str(h))
    fig.circle(Time[0:len(priv_mpc)], priv_mpc[0:len(priv_mpc)], color="forestgreen", legend_label="p mpc "+str(h))

    # Save the plot as a JPG file
    export.export_png(fig, filename="images/plot_priv.png")




def plot_utility(Data_real, Data_obf, Data_MPC, Data_MPC_FLAIR2, h):
    """Plots the utility graph

    Args:
        Data_real (_type_): _description_
        Data_obf (_type_): _description_
        Data_MPC (_type_): _description_
        Data_MPC_FLAIR2 (_type_): _description_
        h (_type_): _description_
    """
    privk_obf = Data_obf['util']
    Time = np.array([30*i for i in range(privk_obf.shape[0])])


    fig = bokeh.plotting.figure(width = 900, height = 450)
    fig.line(Time, privk_obf, line_color='orange', legend_label="geo I")
    fig.circle(Time, privk_obf, color='orange', legend_label="geo I")
    priv_mpc_FLAIR = Data_MPC_FLAIR2.util
    fig.line(Time[0:len(priv_mpc_FLAIR)], priv_mpc_FLAIR, line_color="darkviolet", legend_label="p mpc FLAIR"+str(h))
    fig.circle(Time[0:len(priv_mpc_FLAIR)], priv_mpc_FLAIR[0:len(priv_mpc_FLAIR)], color="darkviolet", legend_label="p mpc FLAIR"+str(h))
    priv_mpc = Data_MPC.util
    fig.line(Time[0:len(priv_mpc)], priv_mpc, line_color="forestgreen", legend_label="p mpc "+str(h))
    fig.circle(Time[0:len(priv_mpc)], priv_mpc[0:len(priv_mpc)], color="forestgreen", legend_label="p mpc "+str(h))
    fig.xaxis.axis_label = "time [s]"
    fig.yaxis.axis_label = "utility loss"

    # Save the plot as a png file
    export.export_png(fig, filename="images/plot_util.png")



def plot_whole_traj(x_coords, y_coords, tk, file_name, Data_mpc_FLAIR, Data_obf, Data_mpc, h):
    """Plots the whole trajectory, as well as the obfuscated trajectories from Geo-I, p mpc-H, and our new model

    Args:
        x_coords (_type_): _description_
        y_coords (_type_): _description_
        tk (_type_): _description_
        file_name (_type_): _description_
    """
    T = np.array([tk*i for i in range(len(x_coords))])
    non_zero_indices = np.nonzero(np.logical_and(x_coords != 0, y_coords != 0))
    x_non_zero = x_coords[non_zero_indices]
    y_non_zero = y_coords[non_zero_indices]
    T = T[non_zero_indices]
    fig = bokeh.plotting.figure(min_width=900, min_height=450)
    fig.line(x_non_zero, y_non_zero, line_color='navy', legend_label="p")
    fig.circle(x_non_zero, y_non_zero, color='navy', legend_label="p")
    fig.circle(Data_obf['x'], Data_obf['y'], color='orange', legend_label="p obf")
    fig.circle(Data_mpc.x, Data_mpc.y, color='forestgreen', legend_label="p mpc " + str(h))
    fig.circle(Data_mpc_FLAIR.x, Data_mpc_FLAIR.y, color="darkviolet", legend_label="p mpc FLAIR " + str(h))
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "y"
    export.export_png(fig, filename="images/" + file_name +".png")


def calculate_percentages(priv, priv_geo, priv_mpc, priv_flair, privacy_limit):
    """ Returns the different statistics we're gonna use in the pdf sum up

    Args:
        priv (array): The array of privacy of the initial traj
        priv_geo (_type_): _description_
        priv_mpc (_type_): _description_
        priv_flair (_type_): _description_
        privacy_limit (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(priv)  # Total number of elements in the arrays

    #percentage of time priv_flair is higher/lower than priv_geo
    flair_higher = sum(priv_flair > priv_geo) / n * 100
    flair_lower = sum(priv_flair < priv_geo) / n * 100

    #percentage of time priv_mpc is higher/lower than priv_geo
    mpc_higher = sum(priv_mpc > priv_geo) / n * 100
    mpc_lower = sum(priv_mpc < priv_geo) / n * 100



    #percentage of time each array is below the privacy_limit
    priv_below_limit = sum(priv < privacy_limit) / n * 100
    geo_below_limit = sum(priv_geo < privacy_limit) / n * 100
    mpc_below_limit = sum(priv_mpc < privacy_limit) / n * 100
    flair_below_limit = sum(priv_flair < privacy_limit) / n * 100

    return flair_higher, flair_lower, mpc_higher, mpc_lower, priv_below_limit, geo_below_limit, mpc_below_limit, flair_below_limit

###################################################### CREATE PDF ########################################################

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def pdf(parameters, statistics):
    """Creates the summary pdf file

    Args:
        parameters (_type_): _description_
        statistics (array): The array created by calculate_percentages
    """
    images1 = parameters['images1']
    images2 = parameters['images2']
    user_name = parameters['user']
    traj_number = parameters['trajectory_index']
    # Create a new PDF file
    pdf_file = f"Sum_up_{parameters['dataset']}_{user_name}_{traj_number}_h{parameters['horizon']}.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    #title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(235, 750, "Trajectory")
    # Set the font and font size
    c.setFont("Helvetica", 12)
    # Add images to the PDF
    x1 = 160
    y1 = 500
    for image in images1:
        c.drawImage(f"images/{image}", x1, y1, width=300, height=225)
        y1 -= 225

    #new page
    c.showPage()
    c.setFont("Helvetica-Bold", 24)
    c.drawString(235, 750, "Privacy / Utility")
    x2 = 160
    y2 = 500
    for image in images2:
        c.drawImage(f"images/{image}", x2, y2, width=350, height=237.5)
        y2 -= 250
    # Add text block with parameters and statistics
    text = f"Parameters:\n\n"
    for key, value in parameters.items():
        text += f"{key}: {value} <br />"
    text += f"\nStatistics:\n\n"
    text += f"Flair Higher than geo-I: {statistics[0]}\n"
    text += f"Flair Lower than geo-I: {statistics[1]}\n"
    text += f"MPC Higher than geo-I: {statistics[2]}\n"
    text += f"MPC Lower than geo-I: {statistics[3]}\n"
    text += f"Priv Below Limit: {statistics[4]}\n"
    text += f"Geo Below Limit: {statistics[5]}\n"
    text += f"MPC Below Limit: {statistics[6]}\n"
    text += f"Flair Below Limit: {statistics[7]}\n"

    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, 50, "Summary Information")
    c.setFont("Helvetica", 12)
    c.drawString(50, 100, text)

    # Save and close the PDF file
    c.save()
    
    
