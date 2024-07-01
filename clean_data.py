from PostGrid import *

"""
    Configuration de l'etude
"""
case = 'C:/Users/G40400404/Desktop/Flow measurement - simulation 3D PTV/Lateral/1 Hz - 0.5 inches/EXP/export_data/'

# Geometrie - Parametres du cylindre et deplacements
R = 14.3/2
DISP_LENGTH = 0.00635 / 2
f = 1
phase = 1.884955592153876


# Calcul des parametres du deplacement impose
T = 1 / f
omega = 2 * np.pi * f
U0 = omega * DISP_LENGTH

# Parametre acquisition
N_images = 500
dt = 1/400

"""
    lecture du cas 
    Calcul des parametres de la grille
    Nettoyage des donnees
    
"""
