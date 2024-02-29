"""
PSO (Particle Swarm Optimization) Algorithm
Streamlit App
"""
import os
os.chdir('C:/Users/lebre/OneDrive/Bureau/PythonS8/PSO_Python')
import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad # for the automatic calculation of gradients
import streamlit as st # app interface
import imageio
import Functions as f
from OptTestFunctions import Rastrigin, Sphere


def PSO_2d(function_choice,N=1,S=100,nbr_iter_max=50,w=0.7,c1=2,c2=2,ub=5.12,fps=8):
    """ In 2d : Create plots and a GIF of the convergence of the PSO algorithm with chosen parameters """

    # Function choice
    function_available= {"Rastrigin":Rastrigin,"Sphere":Sphere}
    Function = function_available.get(function_choice)
    if Function is None:
        print("Function not recognized. Exiting the program.")
        exit()


    # Search space
    min_position = -ub * np.ones(shape=(S, 1))
    max_position = ub * np.ones(shape=(S, 1))

    # Particles at stage 0
    X = min_position + np.random.rand(S,N) * (2 * max_position)
    # Personal bests at stage 0
    Pb = np.copy(X)
    # Velocities at stage 0
    V = np.empty((S,N))
    # Global best at stage 0
    Gb_image = Function(X[0,],N)
    Gb = np.copy(X[0,])

    fig, ax = plt.subplots() # fig creation
    

    # Final loop with a stopping criterion based on a maximum number of iterations
    # & Plot the particles
    nbr_iter = 1

    while nbr_iter < nbr_iter_max:
        
        # parameters initialization
        # w = 0.9 - nbr_iter * (0.9 - 0.4) / nbr_iter_max
        # c1 = 2 - nbr_iter *(2 -0.1) / nbr_iter_max
        # c2 = 0.1 + nbr_iter *(2 -0.1) / nbr_iter_max

        # Reinitialization of V, X, Pb, Gb
        V = f.update_V(w, c1, c2, X, V, Pb, Gb)
        X = f.update_X(X, V)
        X = f.Penalty(X, N, S, ub)
        Pb = f.update_Pb(X, Pb, S, Function, N)
        Gb = f.update_Gb(Gb, Pb, S, Function, N)

        nbr_iter += 1

        # Plotting
        ax.clear()
        f_X=np.ones(shape=(S, 1))
        f_Pb=np.ones(shape=(S, 1))
        f_Gb=np.ones(shape=(1, 1))

        for s in range(S):
            f_X[s,]=Function(X[s,], N)
            f_Pb[s,]=Function(Pb[s,], N)

        # Plotting X, Pb, Gb
        ax.plot(X, f_X,'ko',label='Particles')
        ax.plot(Pb, f_Pb,'bo',label='Personal Bests')
        ax.plot(Gb, Function(Gb, N),'go',label='Global Best')
        # global minima
        if function_choice=="Rastrigin":
            ax.plot(0,0,'ro',label='Global Minimum')
        else:
            ax.plot(1,-4,'ro',label='Global Minimum')

        # plot options
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # title
        ax.set_title(f'Iteration {nbr_iter}')
        # activate legend
        ax.legend()
        # export figures
        filename = 'Images/iteration'+str(nbr_iter)+'.png'
        plt.savefig(filename)

    # GIF creation
    plots = []
    for i in range(1,nbr_iter_max):
        filename='Images/iteration'+str(i+1)+'.png'
        plots.append(imageio.imread(filename))
    imageio.mimsave('Gif/PSO_convergence.gif', plots, fps=fps)

    # Results
    y_Gb = Function(Gb,N)
    gradient = grad(Function)
    grad_result = gradient(Gb, N)

    Gb=np.round(Gb,2)
    y_Gb=np.round(y_Gb,2)
    grad_result=np.round(grad_result,2)

    return Gb, y_Gb, grad_result



def PSO_3d(function_choice,N=2,S=100,nbr_iter_max=10,w=0.7,c1=2,c2=2,ub=5.12,fps=8):
    """ In 3d : Create plots and a GIF of the convergence of the PSO algorithm with chosen parameters """

    # Function choice
    function_available= {"Rastrigin":Rastrigin,"Sphere":Sphere}
    Function = function_available.get(function_choice)
    if Function is None:
        print("Function not recognized. Exiting the program.")
        exit()


    # Search space
    min_position = -ub * np.ones(shape=(S, 1))
    max_position = ub * np.ones(shape=(S, 1))
    
    # Particles at stage 0
    X = min_position + np.random.rand(S,N) * (2 * max_position)
    
    # Personal bests at stage 0
    Pb = np.copy(X)
    
    # Velocities at stage 0
    V = np.empty((S,N))
    
    # Global best at stage 0
    Gb_image = Function(X[0,],N)
    Gb = np.copy(X[0,])
    
    # figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Final loop with a stopping criterion based on a maximum number of iterations
    # & Plot the particles
    nbr_iter = 1
    
    while nbr_iter < nbr_iter_max:
        
        # parameters initialization
        # w = 0.9 - nbr_iter * (0.9 - 0.4) / nbr_iter_max
        # c1 = 2 - nbr_iter *(2 -0.1) / nbr_iter_max
        # c2 = 0.1 + nbr_iter *(2 -0.1) / nbr_iter_max

        # Reinitialization of V, X, Pb, Gb
        V = f.update_V(w, c1, c2, X, V, Pb, Gb)
        X = f.update_X(X, V)
        X = f.Penalty(X, N, S, ub)
        Pb = f.update_Pb(X, Pb, S, Function, N)
        Gb = f.update_Gb(Gb, Pb, S, Function, N)

        nbr_iter += 1

        # Plotting
        ax.clear()

        f_X=np.ones(shape=(S, 1))
        f_Pb=np.ones(shape=(S, 1))
        
        for s in range(S):
            f_X[s,0]=Function(X[s,], N)
            f_Pb[s,0]=Function(Pb[s,], N)
        
        # Plotting X, Pb, Gb
        X_data = (X[:,0],X[:,1], f_X)
        Pb_data = (Pb[:,0],Pb[:,1], f_Pb)
        Gb_data = (Gb[0],Gb[1], Function(Gb, N))
        
        ax.scatter(*X_data, color='black',label='Particles')
        ax.scatter(*Pb_data, color='blue',label='Personal Bests')
        ax.scatter(*Gb_data, color='green',s=100,label='Global Best')
        # global minima
        if function_choice =="Rastrigin":
            ax.scatter(0, 0,0,color='r',s=200,label='Global Minimum')
        else:
            ax.scatter(1, 1,-4,color='r',s=200,label='Global Minimum')
        # plot options
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # plot scales
        ax.set_xlim([-ub,ub])
        ax.set_ylim([-ub,ub])
        ax.set_zlim([-5,100])
        # title
        ax.set_title(f'Iteration {nbr_iter}')
        # activate legend
        ax.legend()
        # export figures
        filename = 'Images/iteration'+str(nbr_iter)+'.png'
        plt.savefig(filename)

    # GIF creation
    plots = []
    for i in range(1,nbr_iter_max):
        filename='Images/iteration'+str(i+1)+'.png'
        plots.append(imageio.imread(filename))
    imageio.mimsave('Gif/PSO_convergence.gif', plots, fps=fps)
    
    # Results
    y_Gb = Function(Gb,N)
    gradient = grad(Function)
    grad_result = gradient(Gb, N)

    Gb=np.round(Gb,2)
    y_Gb=np.round(y_Gb,2)
    grad_result=np.round(grad_result,2)

    return Gb, y_Gb, grad_result


def plotting_fct(function_choice,ub=5.12,N=2):
    """ Plot a 3d function """

    # Function choice
    function_available= {"Rastrigin":Rastrigin,"Sphere":Sphere}
    Function = function_available.get(function_choice)
    if Function is None:
        print("Function not recognized. Exiting the program.")
        exit()

    if N==2: # 3d
        # x,y,z
        x = np.linspace(-ub,ub, 400)
        y = np.linspace(-ub,ub, 400)
        x,y= np.meshgrid(x,y)
        fct_vectorized= np.vectorize(lambda x, y: Function([x,y],N))
        z =fct_vectorized(x,y)

        # figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')

        # global minima
        if function_choice =="Rastrigin":
            ax.scatter(0, 0,0,color='r')
        else:
            ax.scatter(1, 1,-4,color='r')

        # plot options
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else: # 2d
        # x,y
        x = np.linspace(-ub,ub, 400)
        fct_vectorized= np.vectorize(lambda x : Function([x],N))
        y =fct_vectorized(x)

        # figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x,y)

        # global minima
        if function_choice=="Rastrigin":
            ax.plot(0,0,'ro')
        else:
            ax.plot(1,-4,'ro')

        # plot options
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


    # title
    title= function_choice + ' function'
    ax.set_title(title)
    
    filename = 'Images/function_plot.png'
    plt.savefig(filename)

# Streamlit app
def st_app():
    """ Display the convergence of the PSO algorithm with chosen parameters on a Streamlit application """

    # page config
    st.set_page_config(page_title="PSO")
    c = 0
    # sidebar
    st.sidebar.header("Parameters")
    c+=1 # o that each widget receives a unique key
    function_name =st.sidebar.selectbox("Function Name",["Rastrigin","Sphere"], key=c)
    c += 1
    N = st.sidebar.slider("Number of dimensions", min_value=1, max_value=2, value=2,key=c)
    c += 1
    S = st.sidebar.slider("Number of particles", min_value=10, max_value=100, value=50,key=c)
    c+=1
    n_I = st.sidebar.slider("Number of iterations", min_value=10, max_value=500, value=50,key=c)
    c+=1
    w = st.sidebar.slider("w (inertia coefficient)", min_value=0.0, max_value=1.0, value=0.5,key=c)
    c+=1
    c1 = st.sidebar.slider("c1 (cognitive coefficient)", min_value=0.0, max_value=4.0, value=2.0,key=c)
    c+=1
    c2 = st.sidebar.slider("c2 (social coefficient)", min_value=0.0, max_value=4.0, value=2.0,key=c)
    c+=1
    fps = st.sidebar.slider("FPS (frames per second)", min_value=0.0, max_value=50.0, value=8.0,key=c)
    c+=1
    
    st.title("Particle Swarm Optimization")
    st.subheader("Visualization")

    if st.sidebar.button('Launch'): # Gif of the convergence
        if N == 1: # plot in 2d
            Gb,y_Gb,grad_result = PSO_2d(function_choice=function_name,N=1,S=S,nbr_iter_max=n_I,w=w,c1=c1,c2=c2,fps=fps)
        else: # plot in 3d
            Gb,y_Gb,grad_result = PSO_3d(function_choice=function_name,N=2,S=S,nbr_iter_max=n_I,w=w,c1=c1,c2=c2,fps=fps)

        st.image('Gif/PSO_convergence.gif')

        # Results
        st.success(f"Global Best: {Gb}")
        st.success(f"Global Best image: {y_Gb}")
        st.success(f"Gradient result: {grad_result}")
    
    
    if st.sidebar.button('Function Graph'): # Plot the function in 3d
        plotting_fct(function_choice=function_name,ub=5.12,N=N)
        st.image('Images/function_plot.png')

    st.info("Parameters")
    parameters = {'Parameter': ['Function','Number of dimensions','Number of particles', 'Number of iterations', 'c1 (cognitive coefficient)', 'c2 (social coefficient)', 'w (inertia coefficient)','FPS'],
              'Value': [function_name,N,S, n_I, c1, c2, w, fps]}

    # Display parameters
    df = pd.DataFrame(parameters)
    df = df.set_index('Parameter')
    st.table(df)

    st.subheader("Principles")
    st.write("""Particle Swarm Optimization (PSO) is a computational method that optimizes a problem by iteratively trying
                to improve a candidate solution with regard to a given measure of quality. It simulates the social behavior
                of birds within a flock or fish schooling, where each particle represents a potential solution. PSO moves
                these particles around in the search-space according to simple mathematical formulae over the particle's
                position and velocity.""")
    st.write("This equation will be composed of these following coefficients which, together, build the PSO algorithm as we know it: ")
    st.markdown('''
            -  x_{i} \: position of the particle i
            -  v_{i} \: velocity of the particle i
            -  c_1 \: cognitive coefficient influence
            -  c_2 \: social coefficient influence 
            -  w \: inertia weight
            -  r_{i} \: random number uniformly  distributed
            ''')
    st.write('The position and velocity of a particle are updated as follows:')
    st.latex(r'''v_{i}^{t+1} = wv_{i}^{t} + c_1 r_1 (Pbest_{i}^{t} - x_{i}^{t}) + c_2 r_2 (Gbest^{t} - x_{i}^{t})''')
    st.latex(r'''x_{i}^{t+1} = x_{i}^{t} + v_{i}^{t+1}''')
    st.write("""Each particle's movement is influenced by its local best known position and is also
                guided toward the best known positions in the search-space, which are updated as better positions are 
                found by other particles. This process is repeated until a satisfactory solution is found, or a set 
                number of iterations are completed.""")
    
# app launcher
st_app()



