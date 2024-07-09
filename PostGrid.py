import numpy as np
import pandas as pd
import re
import os
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft, fftfreq

def read(file_path):
    # Lire toutes les lignes du fichier
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialiser la liste pour stocker les données
    data_list = []

    # Parcourir les lignes du fichier et collecter les données après la zone des métadonnées
    solution_time = None
    for line in lines:
        if 'SOLUTIONTIME' in line:
            solution_time = float(re.search(r'SOLUTIONTIME\s*=\s*(\d*\.?\d+)', line).group(1))
        #if line.strip().isdigit() or line.strip().startswith('-'):
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
            data_list.append([solution_time] + [val for val in line.split()])

    # Définir les colonnes pour le DataFrame
    columns = ["time", "x", "y", "z", 
               "Velocity u", "Velocity v", "Velocity w", "Velocity |V|", 
               "du/dx", "du/dy", "du/dz", "dv/dx", "dv/dy", "dv/dz", "dw/dx", "dw/dy", "dw/dz", 
               "Vorticity w_x (dw/dy - dv/dz)", "Vorticity w_y (du/dz - dw/dx)", "Vorticity w_z (dv/dx - du/dy)", "|Vorticity|", 
               "Divergence 2D (du/dx + dv/dy)", "Divergence 3D (du/dx + dv/dy + dw/dz)", 
               "Swirling strength 3D (L_2)", "Pressure", "isValid"]
    
    data = pd.DataFrame(data_list, columns=columns)
    
    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col])
    #print(data.columns)
    data['x'] = data['x'] + abs(np.min(data['x']))
    data['y'] = data['y'] + abs(np.min(data['y']))
    data['Vx'] = data['Velocity u']
    data['Vy'] = data['Velocity v']
    data['Vz'] = data['Velocity w']

    data['Velocity'] = data[['Vx', 'Vy', 'Vz']].values.tolist()
    data['position'] = data[['x', 'y', 'z']].values.tolist()
    data['Vorticity'] = data[["Vorticity w_x (dw/dy - dv/dz)", "Vorticity w_y (du/dz - dw/dx)", "Vorticity w_z (dv/dx - du/dy)"]].values.tolist()
    data['gradV'] = np.array([[data['du/dx'].values, data['du/dy'].values, data['du/dz'].values],
                            [data['dv/dx'].values, data['dv/dy'].values, data['dv/dz'].values],
                            [data['dw/dx'].values, data['dw/dy'].values, data['dw/dz'].values]]).T.tolist()
    data['Swirling strength'] = data["Swirling strength 3D (L_2)"]

    final_columns = ['time', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'position', 'Velocity', 'Pressure', 'Vorticity', 'gradV', "Swirling strength",]
    result = data[final_columns]
    return result

def read_data(file_path, dt):
    file_name = file_path.split('/')[-1]  # Obtient le nom du fichier
    match = re.search(r'B(\d+)\.csv', file_name)
    if not match:
        raise ValueError("Le nom de fichier ne correspond pas au format attendu 'B_n.csv'")
    n = int(match.group(1))

    # Computation of SOLUTIONTIME
    solution_time = n * dt

    data = pd.read_csv(file_path, sep = ";")
    data['time'] = solution_time # Adding the time column
    
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col])

    # Adjusting x and y space variables to start at 0
    data['x'] = data['x'] + abs(np.min(data['x']))
    data['y'] = data['y'] + abs(np.min(data['y']))
    
    data['Vx'] = data['Velocity u']
    data['Vy'] = data['Velocity v']
    data['Vz'] = data['Velocity w']
    data['Velocity'] = data[['Vx', 'Vy', 'Vz']].values.tolist()
    data['position'] = data[['x', 'y', 'z']].values.tolist()
    data['Vorticity'] = data[["Vorticity w_x (dw/dy - dv/dz)", "Vorticity w_y (du/dz - dw/dx)", "Vorticity w_z (dv/dx - du/dy)"]].values.tolist()
    data['gradV'] = np.array([[data['du/dx'].values, data['du/dy'].values, data['du/dz'].values],
                              [data['dv/dx'].values, data['dv/dy'].values, data['dv/dz'].values],
                              [data['dw/dx'].values, data['dw/dy'].values, data['dw/dz'].values]]).T.tolist()
    data['Swirling strength'] = data["Swirling strength 3D (L_2)"]

    # Defining final columns
    final_columns = ['time', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'position', 'Velocity', 'Pressure', 'Vorticity', 'gradV', "Swirling strength"]
    result = data[final_columns]
    return result

def read_all_timesteps(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()

    all_data = []
    n = 0
    for filename in files:
        print(f'Reading timestep {n}')
        file_path = os.path.join(directory, filename)
        timestep_data = read(file_path)
        all_data.append(timestep_data)
        n+=1

    combined_data = pd.concat(all_data, ignore_index=True)

    return combined_data

def read_all_timesteps2(directory, dt):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()

    all_data = []
    n = 0
    for filename in files:
        print(f'Reading timestep {n}')
        file_path = os.path.join(directory, filename)
        timestep_data = read_data(file_path, dt)
        all_data.append(timestep_data)
        n+=1

    combined_data = pd.concat(all_data, ignore_index=True)

    return combined_data


def read_csv_to_dataframe(file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Display the first few rows of the DataFrame to confirm it's loaded correctly
        print(df.head())

        # Return the DataFrame
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def write_tecplot_csv(data, file_path, n):
    data = data.drop(columns=['time'], errors='ignore')
    with open(file_path, 'w') as file:
        # Écriture de l'entête Tecplot
        file.write(f"TITLE = \"B{n:04}\"\n")
        file.write("VARIABLES = " + ", ".join([f"\"{col}\"" for col in data.columns]) + "\n")

        # Écrire la zone de données
        file.write("ZONE T=\"Frame 0\", I=38, J=30, K=10, F=POINT\n")
        file.write("STRANDID=1, SOLUTIONTIME=0\n")

        # Écriture des données
        for index, row in data.iterrows():
            file.write(" ".join(row.astype(str)) + "\n")


"""
    Filtering / sampling functions
        ** Filter functions :
        - filter_by_time : Filter the data set to extract the data for a given time value
        - remove_points_inside_cylinder : Filter the given fluid dataset to remove the fluid in the given cylinder
        - clean_data : Same function as remove_points_inside_cylindersb but does it for every timestep

        ** Sampling functions :
        - sample_over_line : Get the fields values over a line by inteporlating
        - sample_over_plan : Get fields values on a plan by interpolation on a grid

"""
def filter_by_time(data, t):
    #print(f"Filtering by time: {t}")
    filtered_data = data[data['time'] == t]
    #print(f"Filtered data size: {len(filtered_data)}")
    return filtered_data

def clip_in_volume(data, x_range, y_range, z_range):
    # Apply conditions for each axis
    condition = (
        (data['position'].apply(lambda pos: pos[0]) >= x_range[0]) & (data['position'].apply(lambda pos: pos[0]) <= x_range[1]) &
        (data['position'].apply(lambda pos: pos[1]) >= y_range[0]) & (data['position'].apply(lambda pos: pos[1]) <= y_range[1]) &
        (data['position'].apply(lambda pos: pos[2]) >= z_range[0]) & (data['position'].apply(lambda pos: pos[2]) <= z_range[1])
    )

    # Filter data based on the defined condition
    clipped_data = data[condition]
    print(f"Clipped data size: {len(clipped_data)}")

    return clipped_data


def sample_over_line(domain, field_name, a, b, line_axis):
    '''
    Extrait les données d'un champ précisé (field_name) le long d'une ligne dont deux points sont fournies (a et b)

    Parameters
    ----------
    domain : Pyvista NP Array
        Domaine sur lequel est défini le champ (fluid, boundary..)
    field_name : str
        Nom du champ à extraire
    a : liste
        Point 1 appartenant à la ligne
    b : liste
        2eme point appartenant à la ligne
    line_axis : str
        Axe // à la ligne :
            Ligne d'axe x : line_axis = "x"
            Ligne d'axe y : line_axis = "y"
            Ligne d'axe z : line_axis = "z"

    Returns
    -------
    Coord : la liste des coordonnées sur la ligne
    field: liste des valeurs du champ sur cette ligne

    '''
    line = pv.Line(pointa=a, pointb=b, resolution=50)
    sampled_data = domain.probe(line)
    if line_axis == "x":
        coord = sampled_data.points[:, 0]
    if line_axis == "y":
        coord = sampled_data.points[:, 1]
    if line_axis == "z":
        coord = sampled_data.points[:, 2]
    field = sampled_data[field_name]
    return(coord, field)


def slice_orthogonal(domain, normal, origin):
    slices = domain.slice(normal=normal, origin=origin)
    return(slices)

def sample_over_plan(domain, normal, origin, field_name):

    plan = slice_orthogonal(domain, normal, origin)

    geo = plan.points.tolist()

    geo = np.array(geo)
    x = [geo[i][0] for i in range(len(geo))]
    if normal == [0,1,0]:
        y = [geo[i][2] for i in range(len(geo))]
    if normal == [0,0,1]:
        y = [geo[i][1] for i in range(len(geo))]
    plan = plan.probe(geo)
    field = plan[field_name]

    return(x, y, field)

def remove_points_inside_cylinder(df_fluid, df_cylinder, R):
    # Interpolate cylinder centers for each fluid data y-value
    x_interp = np.interp(df_fluid['y'], df_cylinder['y'], df_cylinder['x'])
    z_interp = np.interp(df_fluid['y'], df_cylinder['y'], df_cylinder['z'])

    radial_dist = (df_fluid['x'] - x_interp)**2 + (df_fluid['z'] - z_interp)**2 # Compute the distance squared from the cylinder axis

    outside = radial_dist >= R**2 # Identify points outside the cylinder

    return df_fluid[outside]


def clean_data(data_fluid, cylinders, R):
    time = cylinders['time'].tolist()
    time = np.unique(time)
    fluid_time = np.unique(data_fluid['time'].tolist())
    cleaned_data = []
    for t in fluid_time:
        if t in time:
            fluid_t = filter_by_time(data_fluid, t) # fixed grid data
            cylinder = filter_by_time(cylinders, t)
            fluid_t = remove_points_inside_cylinder(fluid_t, cylinder, R)
        else:
            fluid_t = filter_by_time(data_fluid, t)

        cleaned_data.append(fluid_t)  # Append the cleaned data for this timestep to the list
    
    data_fluid_cleaned = pd.concat(cleaned_data, ignore_index=True)  # Concatenate all data frames into one
    return data_fluid_cleaned


"""
    Grids functions
"""
def compute_grid_spacing(data):
    x, y, z = np.unique(data['x']), np.unique(data['y']), np.unique(data['z'])
    dx = np.max(np.diff(x))
    dy = np.max(np.diff(y))
    dz = np.max(np.diff(z))

    return(dx, dy, dz)

def construct_grid(df):
    unique_x = df['x'].unique()
    unique_y = df['y'].unique()
    unique_z = df['z'].unique()
    nx = len(unique_x)
    ny = len(unique_y)
    nz = len(unique_z)

    points = df[['x', 'y', 'z']].values
    velocity = df['Velocity'].values
    vorticity = df['Vorticity'].values
    pressure = df['Pressure'].values
    Sw = df['Swirling strength'].values
    Vx = df['Vx'].values
    Vy = df['Vy'].values
    Vz = df['Vz'].values

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, nz]  # nx, ny, nz doivent être définis

    fields = [velocity, vorticity, pressure, Sw, Vx, Vy, Vz]
    fields_name = ['Velocity', 'Vorticity', 'Pressure', 'Swirling strength', 'Vx', 'Vy', 'Vz']
    for i in range(len(fields)):
        grid.point_data[fields_name[i]] = fields[i].tolist()

    # Extraire les composantes des dérivées de gradV
    gradV = np.array([np.array(matrix).reshape(3, 3) for matrix in df['gradV'].values])

    grad_components = {
        'du_dx': gradV[:, 0, 0],
        'du_dy': gradV[:, 0, 1],
        'du_dz': gradV[:, 0, 2],
        'dv_dx': gradV[:, 1, 0],
        'dv_dy': gradV[:, 1, 1],
        'dv_dz': gradV[:, 1, 2],
        'dw_dx': gradV[:, 2, 0],
        'dw_dy': gradV[:, 2, 1],
        'dw_dz': gradV[:, 2, 2],
    }

    for component_name, component_data in grad_components.items():
        grid.point_data[component_name] = component_data.tolist()

    # Ajouter les champs supplémentaires pour les fluctuations de vitesse et le tenseur de Reynolds
    additional_fields = [
        df['Vx_mean'].values, df['Vy_mean'].values, df['Vz_mean'].values,
        df['Vx_p'].values, df['Vy_p'].values, df['Vz_p'].values,
        df['Rxx'].values, df['Rxy'].values, df['Rxz'].values,
        df['Ryx'].values, df['Ryy'].values, df['Rzx'].values,
        df['Rzx'].values, df['Rzy'].values, df['Rzz'].values
    ]

    additional_fields_names = [
        'Vx_mean', 'Vy_mean', 'Vz_mean', 'Vx_p', 'Vy_p', 'Vz_p',
        'Rxx', 'Rxy', 'Rxz', 'Ryx', 'Ryy', 'Ryz', 'Rzx', 'Rzy','Rzz'
    ]

    for name, data in zip(additional_fields_names, additional_fields):
        grid.point_data[name] = data.tolist()

    grid = grid.delaunay_3d(alpha=0.0, tol=0.0001, offset=2.5, progress_bar=True)

    return grid

def construct_grid_old(df):
    unique_x = df['x'].unique()
    unique_y = df['y'].unique()
    unique_z = df['z'].unique()
    nx = len(unique_x)
    ny = len(unique_y)
    nz = len(unique_z)

    points = df[['x', 'y', 'z']].values
    velocity = df['Velocity'].values
    vorticity = df['Vorticity'].values
    pressure = df['Pressure'].values
    Sw = df['Swirling strength'].values
    Vx = df['Vx'].values
    Vy = df['Vy'].values
    Vz = df['Vz'].values

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, nz]  # nx, ny, nz doivent être définis

    fields = [velocity, vorticity, pressure, Sw, Vx, Vy, Vz]
    fields_name = ['Velocity', 'Vorticity', 'Pressure', 'Swirling strength', 'Vx', 'Vy', 'Vz']
    for i in range(len(fields)):
        grid.point_data[fields_name[i]] = fields[i].tolist()

    # Extraire les composantes des dérivées de gradV
    gradV = np.array([np.array(matrix).reshape(3, 3) for matrix in df['gradV'].values])

    grad_components = {
        'du_dx': gradV[:, 0, 0],
        'du_dy': gradV[:, 0, 1],
        'du_dz': gradV[:, 0, 2],
        'dv_dx': gradV[:, 1, 0],
        'dv_dy': gradV[:, 1, 1],
        'dv_dz': gradV[:, 1, 2],
        'dw_dx': gradV[:, 2, 0],
        'dw_dy': gradV[:, 2, 1],
        'dw_dz': gradV[:, 2, 2],
    }

    for component_name, component_data in grad_components.items():
        grid.point_data[component_name] = component_data.tolist()

    grid = grid.delaunay_3d(alpha=0.0, tol=0.0001, offset=2.5, progress_bar=True)

    return grid


"""
    Usefull functions
"""
def tri_double(X,Y):
    sorted_indices = np.argsort(X)
    X = np.array(X)[sorted_indices]
    Y = np.array(Y)[sorted_indices]
    return(X,Y)

def merge_dataframes(df1, df2, on_columns):
    """
    Fusionne deux DataFrames sur les colonnes communes spécifiées.
    
    Paramètres:
    - df1: Premier DataFrame.
    - df2: Deuxième DataFrame.
    - on_columns: Liste des noms de colonnes sur lesquelles fusionner les DataFrames.
    
    Retourne:
    - DataFrame fusionné.
    """
    merged_df = pd.merge(df1, df2, on=on_columns)
    return merged_df

"""
    Turbulence analysis
        - Velocity field average computation (temporeal average)
        - Velocity fluctuations computation
        - Reynolds tensor computation
        - PSD computation

"""

def calculate_temporal_mean(data):
    print('Computing temporel mean of the velocity field')
    grouped = data.groupby(['x', 'y', 'z'])
    mean_velocities = grouped[['Vx', 'Vy', 'Vz']].mean().reset_index()
    mean_velocities.columns = ['x', 'y', 'z', 'Vx_mean', 'Vy_mean', 'Vz_mean']
    return(mean_velocities)

def calculate_fluctuations(data):
    """
    Calcule les fluctuations temporelles des composantes de vitesse pour chaque point spatial.
    
    Paramètres:
    - data: DataFrame contenant les données avec les colonnes ['time', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']
    
    Retourne:
    - DataFrame avec les fluctuations temporelles des composantes de vitesse pour chaque point spatial
    """
    
    if 'Vx_mean' not in data:
        mean_velocities = calculate_temporal_mean(data) # Calculer la moyenne temporelle
        data = pd.merge(data, mean_velocities, on=['x', 'y', 'z']) # Fusionner les moyennes temporelles avec les données originales

    # Calculer les fluctuations temporelles
    print('Computing velocity temporal fluctuations')
    data['Vx_p'] = data['Vx'] - data['Vx_mean']
    data['Vy_p'] = data['Vy'] - data['Vy_mean']
    data['Vz_p'] = data['Vz'] - data['Vz_mean']
    
    # Sélectionner les colonnes d'intérêt
    fluctuations = data[['time', 'x', 'y', 'z', 'Vx_p', 'Vy_p', 'Vz_p']]
    
    return fluctuations, data

def calculate_reynolds_stress(data):
    """
    Calcule le tenseur des contraintes de Reynolds pour chaque point spatial.
    
    Paramètres:
    - data: DataFrame contenant les données avec les colonnes ['time', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']
    
    Retourne:
    - DataFrame avec les composantes du tenseur des contraintes de Reynolds pour chaque point spatial
    """

    print('Computing Reynolds stress tensor')

    grouped = data.groupby(['x', 'y', 'z']) # Grouper les données par les coordonnées spatiales (x, y, z) et temporel
    
    # Calculer les composantes du tenseur des contraintes de Reynolds
    reynolds_stress = grouped.apply(lambda g: pd.Series({
        'Rxx': (g['Vx_p'] * g['Vx_p']).mean(),
        'Rxy': (g['Vx_p'] * g['Vy_p']).mean(),
        'Rxz': (g['Vx_p'] * g['Vz_p']).mean(),
        'Ryx': (g['Vy_p'] * g['Vx_p']).mean(),
        'Ryy': (g['Vy_p'] * g['Vy_p']).mean(),
        'Ryz': (g['Vy_p'] * g['Vz_p']).mean(),
        'Rzx': (g['Vz_p'] * g['Vx_p']).mean(),
        'Rzy': (g['Vz_p'] * g['Vy_p']).mean(),
        'Rzz': (g['Vz_p'] * g['Vz_p']).mean()
    })).reset_index()
    
    return reynolds_stress



def calculate_psd(x, V, V_fluct):
    """
    Calcule la densité spectrale de puissance (PSD) du champ V(x) le long de la ligne x.
    
    :param x: array-like, les coordonnées x
    :param V: array-like, les valeurs du champ V en x
    :return: freq, psd, les fréquences spatiales et la PSD correspondante
    """
    # Assurer que les données sont uniformément espacées
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Les valeurs de x doivent être uniformément espacées")

    # Calculer la moyenne et les fluctuations de vitesse
    #V_mean = np.mean(V)
    #V_fluct = V - V_mean
    #V_fluct = V_fluct * 1000
    
    V_fft = fft(V_fluct) # Calcul de la transformée de Fourier des fluctuations de vitesse
    
    # Calculer les fréquences spatiales
    N = len(V)
    freq = fftfreq(N, dx)[:N//2]
    
    # Calculer la densité spectrale de puissance (PSD)
    psd = (np.abs(V_fft[:N//2])**2) / (np.trapz(np.abs(V_fft[:N//2]), dx=freq[1] - freq[0])) * np.var(V)


    return freq, psd