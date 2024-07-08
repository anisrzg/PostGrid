from PostGrid import *

"""
    Configuration de l'etude
"""
case = 'C:/Users/G40400404/Desktop/Flow measurement - simulation 3D PTV/Lateral/1 Hz - 0.5 inches/EXP/export_data/'

# Geometrie - Parametres du cylindre et deplacements
R = 14.3/2
DISP_LENGTH = 0.00635 / 2
f = 1.15
phase = 1.884955592153876


# Calcul des parametres du deplacement impose
T = 1 / f
omega = 2 * np.pi * f
U0 = omega * DISP_LENGTH

# Parametre acquisition
N_images = 500
dt = 1/400

SAVE_DIRECTORY = 'C:/Users/G40400404/Desktop/Flow measurement - simulation 3D PTV/Lateral/1 Hz - 0.5 inches/EXP/Post-data/'

"""
    lecture du cas 
    Calcul des parametres de la grille
    Nettoyage des donnees
    
"""
data = read_all_timesteps(case) # Lecture du cas
time = data["time"].tolist()
time = np.unique(time)

# Nettoyage des donnees
cylinders = read_csv_to_dataframe('cylinder_centers.csv')
data = clean_data(data, cylinders, R)

xu, yu, zu = np.unique(data['x']), np.unique(data['y']), np.unique(data['z'])
dx, dy, dz = compute_grid_spacing(data) # Compute grid spacing


"""
    Calcul des grandeurs de la turbulence
    Ajout de ces champs aux grilles.
"""

fluctuations, data = calculate_fluctuations(data) # Calcul des fluctuations de vitesse
Rstress = calculate_reynolds_stress(data) # Calcul du tensor de reynolds
data = merge_dataframes(data, g, ['time', 'x', 'y', 'z'])


"""
    Construction des grilles Pyvista
"""
# Construction des grilles vtk et du block global
dataset = []
n = 0
for t in time:
    df = filter_by_time(data, t)
    grid = construct_grid(df)
    grid.save(f'{SAVE_DIRECTORY}/VTK data/output_{n:05}.vtk')
    dataset.append(grid)
    n+=1
