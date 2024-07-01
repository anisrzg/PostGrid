import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

variables = ['Vx', 'Vy', 'Vz']
times = data['time'].unique()

max_points = data.groupby('time').size().max() # dimension spatiale maximale
snapshots = []
for t in times:
    snapshot = data[data['time'] == t][variables].values
    if snapshot.shape[0] < max_points:
        pad_size = max_points - snapshot.shape[0]
        snapshot = np.pad(snapshot, ((0, pad_size), (0, 0)), mode='constant')
    snapshots.append(snapshot.flatten())
snapshots = np.array(snapshots).T

# Centrage des donnees
mean_data = np.mean(snapshots, axis=1, keepdims=True)
data_centered = snapshots - mean_data

U, Sigma, VT = np.linalg.svd(data_centered, full_matrices=False) # Decomposition SVD
eigenvalues = (Sigma**2) / (data_centered.shape[1] - 1) # Extraction valeurs propres
modes = U # Extration vecteurs propres
coefficients = np.dot(data_centered.T, modes) # Calcul coef de projections sur les modes

# Calcul coefficients prediction
predicted_coefficients = np.dot(data_centered.T, modes)
projection_energy = np.sum(coefficients**2, axis=0)
prediction_energy = np.sum(predicted_coefficients**2, axis=0)


num_modes_to_plot = 9  # Nombre de modes à tracer


"""
    Plot valeurs propres
"""
plt.plot(eigenvalues, 'o-')
plt.xlabel('Numéro du mode')
plt.ylabel('Valeur propre')
plt.title('Valeurs propres de la POD')
plt.show()


"""
    Plot Coefficient des modes POD
"""
num_cols = 3
num_rows = (num_modes_to_plot + num_cols - 1) // num_cols
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))
for i in range(num_modes_to_plot):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].plot(times, coefficients[:, i], color = 'black')
    axs[row, col].set_ylabel('Coefficient Amplitude')
    axs[row, col].set_xlabel('Time')
    axs[row, col].set_title(f'Coefficients POD mode {i}')
    
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flat[j])

fig.suptitle('POD Coefficients for different modes', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuster pour ne pas chevaucher le titre
plt.show()    
     

"""
    Plot Energie modes POD / POD ROM
"""
plt.figure(figsize=(12, 8))
plt.plot(projection_energy, 'o-', label='Projection Energy (POD)')
plt.plot(prediction_energy, 'x-', label='Prediction Energy (POD ROM)')
plt.xlabel('Mode number')
plt.ylabel('Energy')
plt.title('Comparison of Energy Content per Mode')
plt.legend()
plt.show()

"""
    Plot comparaisons coef modes projetes/predit
"""
num_cols = 3
num_rows = (num_modes_to_plot + num_cols - 1) // num_cols
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))

for i in range(num_modes_to_plot):
    ax = axs[i // num_cols, i % num_cols]
    ax.plot(times, coefficients[:, i], 'k-', label='Projeté')  # Ligne continue noire
    ax.plot(times, predicted_coefficients[:, i], 'k--', label='Prédit')  # Ligne pointillée noire
    ax.set_xlabel('Temps')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'POD {i + 1}')
    ax.legend()

for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flat[j])

fig.suptitle('Comparison of predicted/projected POD Coefficients for different modes', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuster pour ne pas chevaucher le titre
plt.show()

"""
    Plot mode 2D plan XZ
"""
num_cols = 3
num_rows = (num_modes_to_plot + num_cols - 1) // num_cols
fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 24))
for i in range(num_modes_to_plot):
    mode_to_plot = i
    mode_reshaped = modes[:, mode_to_plot].reshape((max_points, len(variables)))

    x = data['x'][:max_points]
    y = data['z'][:max_points]

    ax = axs[i // num_cols, i % num_cols]
    contour = ax.tricontourf(x, y, mode_reshaped[:, 0], levels=100, cmap='turbo')
    fig.colorbar(contour, ax=ax, label='Amplitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Isovaleurs du mode POD {mode_to_plot + 1}')

for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flat[j])

plt.tight_layout()
plt.show()   


"""
    Plot mode 2D plan XY
"""
num_cols = 3
num_rows = (num_modes_to_plot + num_cols - 1) // num_cols
fig, axs = plt.subplots(num_rows, num_cols, figsize=(24, 22))
for i in range(num_modes_to_plot):
    mode_to_plot = i
    mode_reshaped = modes[:, mode_to_plot].reshape((max_points, len(variables)))

    x = data['x'][:max_points]
    y = data['y'][:max_points]

    ax = axs[i // num_cols, i % num_cols]
    contour = ax.tricontourf(x, y, mode_reshaped[:, 0], levels=100, cmap='turbo')
    fig.colorbar(contour, ax=ax, label='Amplitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Isovaleurs du mode POD {mode_to_plot + 1}')

for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flat[j])

plt.tight_layout()
plt.show()