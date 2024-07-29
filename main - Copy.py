from PostGrid import *

"""
    Configuration de l'etude
"""
case = 'C:/Users/G40400404/Desktop/Flow measurement - simulation 3D PTV/SBF/Static/5 Hz/Rod measurement/export_data3'

# Geometrie - Parametres du cylindre et deplacements
R = 14.25/2

N_images = 500
dt = 1/2000

SAVE_DIRECTORY = 'C:/Users/G40400404/Desktop/grid-data'

"""
    lecture du cas 
    Calcul des parametres de la grille
    Nettoyage des donnees
    
"""
data = read_all_timesteps(case) # Lecture du cas
time = data["time"].tolist()
time = np.unique(time)



xc1, yc1 = 8.680208547808455, 12.141105683518129
xc2, yc2 = 8.680208547808455, -6.368894316481871

data = remove_points_inside_cylinder2(data, xc1, yc1, R)
data = remove_points_inside_cylinder2(data, xc2, yc2, R)

x = data['x']
z = data['z']

plt.scatter(x, z)
plt.show()

# plot sans interpolation
datat = filter_by_time(data, 80*dt)
x = datat['x']
z = datat['z']
Vx = datat['Vx']
Vy = datat['Vy'] * 1000
Vz = datat['Vz']

# Création de la figure et des axes
fig, ax = plt.subplots()

# Scatter plot pour visualiser les points de données avec des couleurs représentant Vy
scatter = ax.scatter(x, z, c=Vy, cmap='turbo', vmin = 50, vmax = 130)

# Ajout des vecteurs de vitesse avec un quiver plot
#ax.quiver(x, z, Vx, Vz, color='black', scale=2, scale_units='xy')

# Ajout de la barre de couleur
cbar = fig.colorbar(scatter, ax=ax, label='Axial velocity (mm/s)', orientation='horizontal', pad=0.07)

# Configuration des axes
ax.set_xlabel('X')
ax.set_ylabel('Z')

plt.show()

# calcul average grid


def calculate_temporal_mean2(data):
    print('Computing temporal mean of the velocity field')

    # Identifier les colonnes scalaires et vecteurs
    scalar_columns = ['Pressure', 'Vx', 'Vy', 'Vz']
    vector_columns = ['Vorticity']

    # Calculer les moyennes des colonnes scalaires
    grouped = data.groupby(['x', 'y', 'z'])
    scalar_means = grouped[scalar_columns].mean().reset_index()

    # Calculer les moyennes des colonnes vectorielles
    for col in vector_columns:
        scalar_means[col] = grouped[col].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index(drop=True)

    return scalar_means

def calculate_mean_along_y(data):
    print('Computing mean along y of the velocity field')

    # Identifier les colonnes scalaires et vecteurs
    scalar_columns = ['Pressure', 'Vx', 'Vy', 'Vz']
    vector_columns = ['Vorticity']

    # Calculer les moyennes des colonnes scalaires le long de y
    grouped = data.groupby(['x', 'z'])
    scalar_means = grouped[scalar_columns].mean().reset_index()

    # Calculer les moyennes des colonnes vectorielles le long de y
    for col in vector_columns:
        scalar_means[col] = grouped[col].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index(drop=True)

    
    return scalar_means

averaged_data = calculate_temporal_mean2(data)
averaged_datay = calculate_mean_along_y(averaged_data)

x = averaged_datay['x']
z = averaged_datay['z']
Vx = averaged_datay['Vx'] * 1000
Vy = averaged_datay['Vy'] * 1000
Vz = averaged_datay['Vz'] * 1000
VorX, VorY, VorZ = averaged_datay['Vorticity'].apply(lambda v: v[0]), averaged_datay['Vorticity'].apply(lambda v: v[1]), averaged_datay['Vorticity'].apply(lambda v: v[2])
Vor = np.sqrt(VorX**2 + VorY**2 + VorZ**2)

# Création de la figure et des axes
fig, ax = plt.subplots()

scatter = ax.scatter(x, z, c=Vy, cmap='turbo')
#scatter = ax.scatter(x, z, c=Vz, cmap='turbo')

#ax.quiver(x, z, Vx, Vz, color='black', scale=2, scale_units='xy')

cbar = fig.colorbar(scatter, ax=ax, label='$Vy', orientation='horizontal', pad=0.07)

# Configuration des axes
ax.set_xlabel('X')
ax.set_ylabel('Z')

plt.show()