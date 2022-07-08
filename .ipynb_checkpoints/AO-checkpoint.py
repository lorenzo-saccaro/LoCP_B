import h5py
import numpy as np
import pysindy as ps
from sklearn.metrics import r2_score, mean_squared_error

def run_sindy(filename, noise_level, train_frac, sampling_freq=1):
    

    data = h5py.File(filename)
    

    t = np.array(data['times'])
    t = t[::sampling_freq] 
    dt = t[1]-t[0]

    x = np.array(data['x'])
    dx = x[1]-x[0]
    
    u = np.array(data['data'])
    u = u[::sampling_freq]
    
    data.close()
    
    train_sample = int(len(t)*train_frac)
    test_sample = int(len(t)*0.2)
    
    savename = './Results/' + filename.split('.h')[0].split('/')[3] + f'_noise_' + str(noise_level) + '_train_' + str(train_sample) + '_dt_' + str(dt) + '.h5'
    
    results = h5py.File(savename, 'w')
    
    
    phase_noise = np.exp(1j*2*np.pi*np.random.normal(0, noise_level, size=len(t)))
    u = u*phase_noise[:, np.newaxis]
    
    results.create_dataset('phase_noise', data=phase_noise, dtype='complex128')
    
    
    u_dot = ps.FiniteDifference(axis=0)._differentiate(u, t=dt)
    
    ## GET REAL AND IMG PART TO USE WITH SINDY ## 

    train_sample = int(len(t)*train_frac)
    test_sample = int(len(t)*0.2)
    
    u_real = np.real(u)
    u_img = np.imag(u)

    u_dot_real = np.real(u_dot)
    u_dot_img = np.imag(u_dot)

    u_shaped = np.zeros(shape=(len(x), len(t), 3))
    u_shaped[:,:, 0]=u_real.T
    u_shaped[:,:, 1]=u_img.T
    u_shaped[:,:, 2]=np.tile(x,(len(t),1)).T # add x values to pass to u and u_dot (in these way one can introduce spatial potential terms)

    u_dot_shaped = np.zeros(shape=(len(x), len(t), 2))
    u_dot_shaped[:,:, 0]=u_dot_real.T
    u_dot_shaped[:,:, 1]=u_dot_img.T
    
    ## DEFINE CANDIDATES LIBRARY ## 

    # 2nd order poly library
    poly_library = ps.PolynomialLibrary(include_bias=False, degree=2) 

    # PDE library
    library_functions = [lambda x: x]
    library_function_names = [lambda x: x]
    pde_library = ps.PDELibrary(library_functions=library_functions, 
                            function_names=library_function_names, 
                            derivative_order=2, spatial_grid=x, 
                            include_bias=True, is_uniform=True, include_interaction=False)

    # Tensor polynomial library with the PDE library
    tensor_array = [[1, 1]]
    inputs_temp = np.tile([0, 1, 2], 2)
    inputs_per_library = np.reshape(inputs_temp, (2, 3))
    inputs_per_library[1,2] = 0 # only compute PDs for psi_real and psi_img
    inputs_per_library[0] = [2, 2, 2] # only compute polynomial for x input feature


    generalized_library = ps.GeneralizedLibrary(
        [poly_library, pde_library],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    
    
    reshape_train_score = u_dot_shaped[:,:train_sample,:].shape[-3]*u_dot_shaped[:,:train_sample,:].shape[-2], u_dot_shaped[:,:train_sample,:].shape[-1]
    reshape_val_score = u_dot_shaped[:,-test_sample:,:].shape[-3]*u_dot_shaped[:,-test_sample:,:].shape[-2], u_dot_shaped[:,-test_sample:,:].shape[-1]


    threshold_values = [1, 0.1, 0.01, 0.001, 0.0001, 0]
    r2_score_train = []
    r2_score_val = []
    mse_train = []
    mse_val = []
    eq = []


    for thres_val in threshold_values:    

        optimizer = ps.STLSQ(threshold=thres_val, verbose=False)
        model = ps.SINDy(feature_library=generalized_library, optimizer=optimizer, feature_names=['psi_r', 'psi_i', 'x'])
        model.fit(x=u_shaped[:,:train_sample,:], t=dt, x_dot=u_dot_shaped[:,:train_sample,:])

        eq.append(model.equations(precision=16)) 

        r2_score_train.append(model.score(x=u_shaped[:,:train_sample,:], t=dt, x_dot=u_dot_shaped[:,:train_sample,:].reshape(reshape_train_score)))
        r2_score_val.append(model.score(x=u_shaped[:,-test_sample:,:], t=dt, x_dot=u_dot_shaped[:,-test_sample:,:].reshape(reshape_val_score)))

        mse_train.append(model.score(x=u_shaped[:,:train_sample,:], t=dt, x_dot=u_dot_shaped[:,:train_sample,:].reshape(reshape_train_score), metric=mean_squared_error))
        mse_val.append(model.score(x=u_shaped[:,-test_sample:,:], t=dt, x_dot=u_dot_shaped[:,-test_sample:,:].reshape(reshape_val_score), metric=mean_squared_error))

    results.create_dataset('threshold_values', data=threshold_values)
    results.create_dataset('r2_score_train', data=r2_score_train)
    results.create_dataset('r2_score_val', data=r2_score_val)
    results.create_dataset('mse_train', data=mse_train)
    results.create_dataset('mse_val', data=mse_val)
    results.create_dataset('eq', data=eq)    
    
    results.close()