import matplotlib
matplotlib.use('Agg')

import pde
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib import animation

def run_sim(x_s,d,mu,sim_length):
    ## POTENTIAL DEFINITION ##

    ## PARAMETERS

    m = 1 # fixed 
    w = 1 # fixed 
    x_s = x_s # local minima positions +/- , change to vary barrier length
    d_c = m*w**2/(2*x_s) # critical value of d 
    d = d # modulates maximum position and barrier height, negative values -> global minimun on the right 
    # d = 0 symmetric d!=0 asymmetric

    if (d <= -d_c or d >= d_c):
        # d must be lower in absolute value than the critical value
        raise ValueError()

    C = m*w**2/(8*x_s**2)
    x_u = d*(2*x_s**2/(m*w**2)) # maximum position
    delta_V = 4*d*x_s**3/3 # between the two minima 



    def V(x):
        return C*(x**2-x_s**2)**2 - d*(x**3/3-x_s**2*x)

    def V0(x): # rescale potential so that Vmax is at zero
        return V(x)-V(x_u)

    print('barrier height', V(x_u)-V(-x_s))
    print('max position', x_u)
    print('delta V', delta_V)

    x = np.linspace(-10,10,1000)

    plt.plot(x,V0(x))
    plt.ylim(V0(x_s)*1.1, 1)
    plt.xlabel('x')
    plt.ylabel(r'$V_0(x)$')
    
    ## DEFINE EQUATION TO SIMULATE ##

    hbar = 1                   # reduced plank constant
    x0 = -x_s                 # center of initial gaussian wave-packet, start on left minima
    p0 = 0                     # momentum of initial gaussian wave-packet
    mu = mu                   # mu = <delta_(x**2)>
    alpha = 0                  # alpha = <delta_x*delta_p + delta_x*delta_p>

    f_name = f'double_well_xs_{x_s:.3f}_d_{d:.3f}_mu_{mu:.1f}.h5'
    path = './Data/Simulations/'
    movie_name = f'double_well_xs_{x_s:.3f}_d_{d:.3f}_mu_{mu:.1f}.mov'
    movie_path = './Data/Movie/'

    eq = pde.PDE({'psi':f'I*0.5*laplace(psi)/{m} - (V(x)-V({x_u}))*I*psi'}, user_funcs={'V':V})

    ## DEFINE SPATIAL GRID ##

    Nx = 1000 # number of spatial points
    x_left = -10
    x_right = 10

    grid = pde.CartesianGrid([(x_left,x_right)], Nx, periodic=False)

    dx = grid.discretization
    x = grid.cell_coords.flatten()

    ## DEFINE INITIAL STATE ##

    # squeezed coherent Gaussian wavepacket 
    psi0 = 1/((2*np.pi*mu)**0.25) * np.exp(-((1-1j*alpha)/(4*mu))*(x-x0)**2 + 1j*p0*(x-x0)/hbar) 
    initial_state = pde.ScalarField(grid=grid, data=psi0, label=r"$|\psi_0|^2$")
    initial_state.plot(scalar='norm_squared') # TODO: ADD LABELS 

    print("Total Probability: ", np.sum(np.abs(initial_state.data)**2)*dx)
    
    ## SOLVE EQUATION ##

    # define storage 
    storage = pde.MemoryStorage() # for movie
    file_storage = pde.FileStorage(filename=path+f_name) # dataset

    # simulation time and step (1e-6 and 1e-5 seems to work, more testing needed to se if less can be used)
    t_range = sim_length
    dt_sim = 1e-5

    # points to store to file
    # N_t_writing = 1000 # can be changed if more points are needed
    dt_writing = 0.01

    # points to use in animation 
    N_t_anim = 1000 # this is enough do not change it 
    dt_anim = t_range/N_t_anim


    solver = pde.ExplicitSolver(eq, scheme="runge-kutta", adaptive=False)
    controller = pde.Controller(solver, t_range=t_range, tracker=['progress', storage.tracker(interval=dt_anim), file_storage.tracker(interval=dt_writing)])
    _ = controller.run(initial_state, dt=dt_sim)
    
    ## APPEND X DATA TO CREATED FILE ## 

    file = h5py.File(path+f_name, 'a')
    file.create_dataset('x', data=x)
    file.close()
    
    ## PLOT SOLUTIONS ##

    pde.plot_kymograph(storage, scalar='norm_squared')
    
    from tqdm import tqdm

    def my_pbar(curr_frame, tot_frame):
        pbar.update(100*curr_frame/tot_frame)
        
    ## GENERATE MOVIE ##

    simulation = storage.data
    times = storage.times

    fig,ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('a.u.')
    title = ax.set_title('', y=1.05)
    line1, = ax.plot(x, V0(x), "k--", label="V(x)")
    line2, = ax.plot(x, np.abs(simulation[0])**2, "b", label=r"$|\psi|^2$")
    plt.legend(loc=1, fontsize=8, fancybox=False)
    plt.ylim(1.05*V0(x_s), np.max(np.abs(simulation)**2)*1.1)


    def init():
        return line1, line2


    def animate(i):
        ax.set_facecolor('white')
        line2.set_data(x, np.abs(simulation[i])**2)
        title.set_text('Time = {0:1.3f}'.format(times[i]))
        return line2


    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=False, frames=np.arange(len(times)))

    writer = animation.FFMpegWriter(fps=30, bitrate=-1)


    print("Generating animation ...")
    pbar = tqdm(total=(len(times)-1)*50) # why times 50 works?
    anim.save(movie_path+movie_name, writer=writer, dpi=150, progress_callback=my_pbar)
    pbar.close()
    print("Done")

