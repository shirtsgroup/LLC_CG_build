#!/usr/bin/env python

import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import mdtraj as md
import random
from scipy import spatial
from scipy.optimize import fsolve

# Some convenience functions
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def sqrt(x):
    return np.sqrt(x)

pi = np.pi

######################################################################################
######### Everything in this section should be in LLC_Membranes already ##############
######################################################################################

# Define the triply periodic minimal surface functions
def SchwarzD(X, period):

    N = 2*pi/period
    
    a = sin(N*X[0]) * sin(N*X[1]) * sin(N*X[2])
    b = sin(N*X[0]) * cos(N*X[1]) * cos(N*X[2])
    c = cos(N*X[0]) * sin(N*X[1]) * cos(N*X[2])
    d = cos(N*X[0]) * cos(N*X[1]) * sin(N*X[2])
    
    return a + b + c + d

def Gyroid(X,period):
    
    N = 2*pi/period
    
    a = sin(N*X[0]) * cos(N*X[1])
    b = sin(N*X[1]) * cos(N*X[2])
    c = sin(N*X[2]) * cos(N*X[0])
    
    return a + b + c

def SchwarzD_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a = N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z)
    b = N*sin(N*x)*cos(N*y)*sin(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z)
    c = N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Gyroid_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a =  N*cos(N*x)*cos(N*y) - N*sin(N*x)*sin(N*z)
    b = -N*sin(N*y)*sin(N*x) + N*cos(N*y)*cos(N*z)
    c = -N*sin(N*y)*sin(N*z) + N*cos(N*z)*cos(N*x)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

# Functions for monomer movement
def Rvect2vect(A, B):
    a = A / np.linalg.norm(A) # to be rotated
    b = B / np.linalg.norm(B) # to rotate to
    
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    
    v_skew = np.array([[    0, -v[2],  v[1]], 
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],    0]])
    
    R = np.identity(3) + v_skew + np.dot(v_skew,v_skew) * (1 - c)/s**2
   
    return np.array(R)

def rotate_coords(coords, R):
    
    pos = np.copy(coords)
    for i in range(np.shape(coords)[0]):
        pos[i,:] = np.dot(R,pos[i,:])
        
    return pos

def translate(coords, from_loc, to_loc):
    
    pos = np.copy(coords)
    direction = to_loc - from_loc
    
    T = np.array([[1,0,0,direction[0]],
                  [0,1,0,direction[1]],
                  [0,0,1,direction[2]],
                  [0,0,0,1           ]])
    
    b = np.ones([1])
    for i in range(pos.shape[0]):
        P = np.concatenate((pos[i,:], b))
        x = np.dot(T,P)
        pos[i,:] = x[:3]
        
    return pos

def get_reference_point(coords, topology, ref_groups):
    
    ref_point = np.zeros([1,3])
    for name in ref_groups:
        ref_idx = topology[topology.name == name].index[0]
        ref_point += coords[ref_idx,:]
        
    ref_point = ref_point[0] / len(ref_groups)
    
    return ref_point

def get_monomer_info(PDB_file):
    
    monomerPDB = md.formats.PDBTrajectoryFile(PDB_file)
    monomerArray = monomerPDB.positions[0,:,:]
    atoms_per_monomer = monomerArray.shape[0]
    top, _ = monomerPDB.topology.to_dataframe()

    return monomerArray, atoms_per_monomer, top
    
def get_monomer_vector(coords, topology, ref_head_groups, ref_tail_groups):
    
    vector_head = np.zeros([1,3])
    for name in ref_head_groups:
        head_idx = topology[topology.name == name].index[0]
        vector_head += coords[head_idx,:]
    
    vector_head = vector_head[0] / len(ref_head_groups)

    vector_tail = np.zeros([1,3])
    for name in ref_tail_groups:
        tail_idx = topology[topology.name == name].index[0]
        vector_tail += coords[tail_idx,:]
    
    vector_tail = vector_tail[0] / len(ref_tail_groups)
    monomer_vector = vector_head - vector_tail

    return monomer_vector

def generate_BCC(n=100, box=9.4, period=9.4, struct='SchwarzD', surf2surf=False, placements=False, r=7.8, n_monomer=614, plot=True, tol=0.01, verbose=False):
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]
    
    if verbose:
        print('Box size is\t %.4f' %(box))
        print('Period is\t %.4f' %(period))
    
    if struct == 'SchwarzD' or struct == 'schwarzD':
        C = SchwarzD(X, period)
    elif struct == 'Gyroid' or struct == 'gyroid':
        C = Gyroid(X, period)
    else:
        raise NameError('Not a valid structure name. Try: SchwarzD or Gyroid')
    
    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -tol < C[i,j,k] < tol:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]
    df = pd.DataFrame(structure, columns=['x','y','z'])
    df['size'] = np.ones(structure.shape[0]) * 0.1
        
    if plot:
        fig = px.scatter_3d(df,x='x',y='y',z='z', opacity=1, size='size')
        fig.show()
    
    if placements:
        count = 0
        monomer_placement = np.zeros([round(n_monomer),3])
        r=r

        while count < n_monomer/2:
            placement = random.randint(0, structure.shape[0] - 1)
            monomer_placement[count,:] = structure[placement,:]
            
            tree = spatial.cKDTree(structure)
            nn = tree.query_ball_point(structure[placement,:], r)
            nn.append(placement)
            structure = np.delete(structure,nn,0)
            
            count += 1
            
            if structure.shape[0] == 0:
                print('No more placements available')
                break
                
        if count < n_monomer/2:
            print('Only %d monomers were placed' %(count))
        else:
            print('Success! All %d monomers were placed' %(n_monomer/2))
            
        return df, monomer_placement
    elif surf2surf:
        return structure
    else:
        return df

######################################################################################
########################### These might have some changes ############################
######################################################################################

# Function for placing monomers
def place_monomers(monomer_placement,shift=10,n_monomer=614,struct='SchwarzD',period=9.4,ref_head_groups=['C18','C19','N','N1','C47','C26','C27','N2','N3','C46'],ref_tail_groups=['C','C1','C44','C45']):
    

    monomerArray, atoms_per_monomer, topology = get_monomer_info('./Dibrpyr14.pdb')
    ref_point = get_reference_point(monomerArray, topology, ref_head_groups)

    total_atoms = round(n_monomer/2 * atoms_per_monomer)
    monolayer1 = np.zeros([total_atoms,3])
    reference_points1 = np.zeros([total_atoms,3])

    for i in range(round(n_monomer/2)):
    
        # Calculate the gradient at new position
        if struct == 'SchwarzD':
            n = SchwarzD_grad(monomer_placement[i,:], period)
        elif struct == 'Gyroid':
            n = Gyroid_grad(monomer_placement[i,:], period)
        else:
            raise NameError('Not a valid structure name. Try: SchwarzD or Gyroid')
        
    
        # Translate the monomer to the origin
        origin_positions = translate(monomerArray, ref_point, [0,0,0])
        monomer_vector = get_monomer_vector(origin_positions, topology, ref_head_groups, ref_tail_groups)
    
        # Rotate the monomer for new position
        R = Rvect2vect(monomer_vector, n)
        rotated_positions = rotate_coords(origin_positions, R)
        ref_point = get_reference_point(rotated_positions, topology, ref_head_groups)
    
        # Shift the placement along the interface
        shifted_placement = monomer_placement[i,:] - shift*n
    
        # Translate the monomer to new position
        translated_positions = translate(rotated_positions, ref_point, shifted_placement)
    
        # Save the new coordinates
        monolayer1[atoms_per_monomer*i:atoms_per_monomer*(i+1)] = translated_positions
        reference_points1[atoms_per_monomer*i:atoms_per_monomer*(i+1)] = get_reference_point(translated_positions, topology, ref_head_groups)
    
    # Place on other side of BCC
    monolayer2 = np.zeros([total_atoms,3])
    reference_points2 = np.zeros([total_atoms,3])
    for i in range(round(n_monomer/2)):
    
        # Calculate the gradient at new position
        if struct == 'SchwarzD':
            n = -SchwarzD_grad(monomer_placement[i,:], period)
        elif struct == 'Gyroid':
            n = -Gyroid_grad(monomer_placement[i,:], period)
        else:
            raise NameError('Not a valid structure name. Try: SchwarzD or Gyroid')
    
        # Translate the monomer to the origin
        origin_positions = translate(monomerArray, ref_point, [0,0,0])
        monomer_vector = get_monomer_vector(origin_positions, topology, ref_head_groups, ref_tail_groups)
    
        # Rotate the monomer for new position
        R = Rvect2vect(monomer_vector, n)
        rotated_positions = rotate_coords(origin_positions, R)
        ref_point = get_reference_point(rotated_positions, topology, ref_head_groups)
    
        # Shift the placement along the interface
        shifted_placement = monomer_placement[i,:] - shift*n
    
        # Translate the monomer to new position
        translated_positions = translate(rotated_positions, ref_point, shifted_placement)
    
        # Save the new coordinates
        monolayer2[atoms_per_monomer*i:atoms_per_monomer*(i+1)] = translated_positions
        reference_points2[atoms_per_monomer*i:atoms_per_monomer*(i+1)] = get_reference_point(translated_positions, topology, ref_head_groups)
    
    bilayer = np.concatenate((monolayer1, monolayer2))
    reference_points = np.concatenate((reference_points1,reference_points2))
    df = pd.DataFrame(bilayer,columns=['x','y','z'])
    df['size'] = np.ones(df.shape[0])
    df['ref_x'] = reference_points[:,0]
    df['ref_y'] = reference_points[:,1]
    df['ref_z'] = reference_points[:,2]
    
    return df

def get_slice(surf,mono,box=94,lower=0,upper=1):
        
    # get slices of the surface
        # Lower bounds
    df1 = surf[surf.x   > lower*box]
    df1 = df1[df1.y     > lower*box]
    df1 = df1[df1.z     > lower*box]
        # Upper bounds
    df1 = df1[df1.x < upper*box]
    df1 = df1[df1.y < upper*box]
    df1 = df1[df1.z < box]
    
    # get slices of the monomers
        # Lower bounds
    df3 = mono[mono.ref_x > lower*box]
    df3 = df3[df3.ref_y   > lower*box]
    df3 = df3[df3.ref_z   > lower*box]
        # Upper bounds
    df3 = df3[df3.ref_x < upper*box]
    df3 = df3[df3.ref_y < upper*box]
    df3 = df3[df3.ref_z < box]
    
    return df1, df3

def generate_color_list(n_plot,atoms_per_monomer=138):
    color_list_inside = []
    for c in px.colors.n_colors((0,0,255),(255,255,255),n_plot):
        for _ in range(atoms_per_monomer):
            color_list_inside += ['rgb' + str(c)]
            
    color_list_outside = []
    for c in px.colors.n_colors((255,0,0),(255,255,255),n_plot):
        for _ in range(atoms_per_monomer):
            color_list_outside += ['rgb' + str(c)]
            
    return color_list_inside, color_list_outside

def plot_full(df1,df3,n_plot,color_list_inside,color_list_outside,surf=True,ins=True,out=True,atoms_per_monomer=138,title='Gyroid',xlims=[0,94],ylims=[0,94],zlims=[0,94],up=[0,0,1],center=[0,0,0],eye=[1.25,1.25,1.25]):
    fig = go.Figure()
    if surf:
        fig.add_trace(
            go.Scatter3d(x=df1.x, y=df1.y, z=df1.z,
                mode='markers', marker_color='gray', marker_line_color='white',marker_line_width=0.1,marker_size=5,
                name='pore'
            ))
    if ins:
        fig.add_trace(
            go.Scatter3d(
                x=df3.head(n_plot*atoms_per_monomer).x,
                y=df3.head(n_plot*atoms_per_monomer).y,
                z=df3.head(n_plot*atoms_per_monomer).z,
                mode='markers', marker_line_width=0.1,
                marker=dict(size=5,line=dict(color='white'),color=color_list_inside),
                legendgroup='group'
            ))
    if out:
        fig.add_trace(
            go.Scatter3d(
                x=df3.tail(n_plot*atoms_per_monomer).x,
                y=df3.tail(n_plot*atoms_per_monomer).y,
                z=df3.tail(n_plot*atoms_per_monomer).z,
                mode='markers', marker_line_width=0.1,
                marker=dict(size=5,line=dict(color='white'),color=color_list_outside),
                legendgroup='group'
            ))

    camera = dict(
            up=dict(x=up[0], y=up[1], z=up[2]),
            center=dict(x=center[0],y=center[1],z=center[2]),
            eye=dict(x=eye[0],y=eye[1],z=eye[2])
        )
        
    fig.update_layout(title=title,
                      width=600, height=600,
                      scene_camera=camera)
    
    fig.update_layout(scene = dict(
                        xaxis = dict(
                             range=xlims),
                        yaxis = dict(
                            range=ylims),
                        zaxis = dict(
                            range=zlims),),
                        width=700,
                        margin=dict(
                        r=10, l=10,
                        b=10, t=10)
                      )
    
    return fig

######################################################################################
################### Here is an example how I use these functions #####################
######################################################################################

df, monomer_placement = generate_BCC(struct='Gyroid',n=200,box=94.0,period=94.0,plot=False,placements=True)
mono = place_monomers(monomer_placement,struct='Gyroid',shift=5,period=94.0,ref_head_groups=['C21','C22','C23','C24'])
df_surf, df_mono = get_slice(df,mono,lower=0,upper=1)
n_plot = round(df_mono.shape[0] / 138 / 2)
c_in, c_out = generate_color_list(n_plot)
fig = plot_full(df_surf,df_mono,round(n_plot),c_in,c_out,eye=[1.25,1.25,1.25],title='')
fig.show()