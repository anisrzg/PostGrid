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

# Construction des grilles vtk et du block global
dataset = []
n = 0
for t in time:
    df = filter_by_time(data, t)
    grid = construct_grid(df)
    grid.save(f'{SAVE_DIRECTORY}/VTK data/output_{n:05}.vtk')
    dataset.append(grid)
    n+=1

"""
    PLOTS / TRAITEMENT
"""

colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']

"""
    Profiles de vitesse
"""

""" Profile Vx(x) sur ligne x """

timesteps = [0, 100, 200, 400] # Time to plot
i = 0
plt.figure(figsize=(8, 8))
for tp in timesteps:
    grid = dataset[tp]
    t = time[tp]
    df = filter_by_time(data, t)
    x = df['x'].tolist()
    #a = [np.min(x), 2, 0.5]
    #b = [np.max(x), 2, 0.5]
    a = [0, 2, 0.5]
    b = [12, 2, 0.5]

    x, V = sample_over_line(grid, "Velocity", a, b, "x")
    V = [v[0] for v in V]

    
    plt.plot(x, V, color = colors[i], label = f't = {time[tp]} s')
    i+=1
plt.xlim(-1, 13)
plt.xlabel('x [mm]')
plt.ylabel('Cross velocity (mm/s)')
plt.grid(linestyle = '--')
plt.legend()
plt.show()




""" Profile Vy(x) sur ligne x """

timesteps = [0, 100, 200, 400] # Time to plot
i = 0
plt.figure(figsize=(8, 8))
for tp in timesteps:
    grid = dataset[tp]
    t = time[tp]
    df = filter_by_time(data, t)
    x = df['x'].tolist()
    a = [0, 2, 0.5]
    b = [12, 2, 0.5]

    x, V = sample_over_line(grid, "Velocity", a, b, "x")
    V = [v[1] for v in V]

    
    plt.plot(x, V, color = colors[i], label = f't = {time[tp]} s')
    i+=1
plt.xlim(-1, 13)
plt.xlabel('x [mm]')
plt.ylabel('Axial velocity (mm/s)')
plt.grid(linestyle = '--')
plt.legend()
plt.show()


"""
    Plot 2D
"""
""" PPlot 2D Vx(x,t) """
a = [0, 5, 0.5]
b = [12, 5, 0.5]

tp = np.arange(0,499,1)
t = [time[i] for i in tp]
x_all = None
Vx_all = []
for i in tp:
    grid = dataset[i]
    x, V = sample_over_line(grid, "Velocity", a, b, "x")
    Vx = [v[0] for v in V]
    if x_all is None:
        x_all = x
    
    Vx_all.append(Vx)

Vx_all = np.array(Vx_all)*1000
plt.figure(figsize=(10, 6))
plt.contourf(x_all, t, Vx_all, levels = 100, cmap='turbo')
plt.colorbar(label='Cross Velocity [mm/s]')
plt.xlabel('x (mm)')
plt.ylabel('Time (s)')
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity for x and t.svg')
plt.show()


""" Plot 2D Vx(y,t) """
a = [5, 0, 0.5]
b = [5, 5, 0.5]

tp = np.arange(0,499,1)
t = [time[i] for i in tp]
x_all = None
Vx_all = []
for i in tp:
    grid = dataset[i]
    x, V = sample_over_line(grid, "Velocity", a, b, "y")
    Vx = [v[0] for v in V]
    if x_all is None:
        x_all = x
    
    Vx_all.append(Vx)

Vx_all = np.array(Vx_all)
plt.figure(figsize=(10, 6))
plt.contourf(x_all, t, Vx_all, levels = 100, cmap='turbo')
plt.colorbar(label='Cross Velocity [mm/s]')
plt.xlabel('y (mm)')
plt.ylabel('Time (s)')
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity for y and t.svg')
plt.show()


""" Plot 2D Vx(z,t) """
a = [0, 5, 1]
b = [12, 5, 1]

tp = np.arange(0,499,1)
t = [time[i] for i in tp]
x_all = None
Vx_all = []
for i in tp:
    grid = dataset[i]
    x, V = sample_over_line(grid, "Velocity", a, b, "x")
    Vx = [v[0] for v in V]
    if x_all is None:
        x_all = x
    
    Vx_all.append(Vx)

Vx_all = np.array(Vx_all) * 1000
plt.figure(figsize=(10, 6))
plt.contourf(x_all, t, Vx_all, levels = 100, cmap='turbo')
plt.colorbar(label='Cross Velocity [mm/s]')
plt.xlabel('z (mm)')
plt.ylabel('Time (s)')
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity for z and t.svg')
plt.show()


""" Plot 2D Vy(x,t) """
a = [0, 5, 1]
b = [12, 5, 1]

tp = np.arange(0,499,1)
t = [time[i] for i in tp]
x_all = None
Vx_all = []
for i in tp:
    grid = dataset[i]
    x, V = sample_over_line(grid, "Velocity", a, b, "x")
    Vx = [v[1] for v in V]
    if x_all is None:
        x_all = x
    
    Vx_all.append(Vx)

Vx_all = np.array(Vx_all) * 1000
plt.figure(figsize=(10, 6))
plt.contourf(x_all, t, Vx_all, levels = 100, cmap='turbo')
plt.colorbar(label='Axial Velocity [mm/s]')
plt.xlabel('x')
plt.ylabel('Time')
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Axial velocity for x and t.svg')
plt.show()



""" Plot 2D mulitple timesteps Vx(x,y) plan XY before the rod is in the field view """
timesteps = [0, 50, 100, 150] # Time to plot

z0 = 0.5

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y , V = sample_over_plan(grid, [0,0,1], [0,0,z0], "Velocity")
        Vx = [v[0] for v in V]
        Vy = [v[1] for v in V]
        
        x, y, V = np.array(x), np.array(y), np.array(V)
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        Vx_grid = griddata((x, y), Vx, (X, Y), method='cubic') * 1000
        Vy_grid = griddata((x, y), Vy, (X, Y), method='cubic') * 1000

        c =  ax.pcolormesh(X, Y, Vx_grid, shading='auto', cmap='turbo', vmin=-45, vmax=45)
        #ax.streamplot(X, Y, Vx_grid, Vy_grid, color='black')
        step = 4
        ax.quiver(X[::step, ::step], Y[::step, ::step], Vx_grid[::step, ::step], Vy_grid[::step, ::step], color='black', scale=20, scale_units='xy')
        ax.set_title(f'$Vx$ for t = {t} s')
    else:
        ax.axis('off') 

cbar = fig.colorbar(c, ax=ax_list, label='$Cross velocity$ (mm/s)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(yu), np.max(yu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity - before rod in view.svg')
plt.show()



""" Plot 2D mulitple timesteps Vx(x,y) plan XY when the rod is in the field view """
timesteps = [160, 210, 330, 380] # Time to plot

z0 = 0.5

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y , V = sample_over_plan(grid, [0,0,1], [0,0, z0], "Velocity")
        Vx = [v[0] for v in V]
        Vy = [v[1] for v in V]
        
        x, y, V = np.array(x), np.array(y), np.array(V)
        xi = np.linspace(x.min(), x.max(), len(xu))
        yi = np.linspace(y.min(), y.max(), len(yu))
        X, Y = np.meshgrid(xi, yi)
        Vx_grid = griddata((x, y), Vx, (X, Y), method='cubic') * 1000
        Vy_grid = griddata((x, y), Vy, (X, Y), method='cubic') * 1000
        
        step = 1
        c =  ax.pcolormesh(X, Y, Vx_grid, shading='auto', cmap='turbo', vmin=-45, vmax=45)
        ax.quiver(X[::step, ::step], Y[::step, ::step], Vx_grid[::step, ::step], Vy_grid[::step, ::step], color='black', scale=20, scale_units='xy')
        #ax.streamplot(X, Y, Vx_grid, Vy_grid, color='black')
        ax.set_title(f'$Vx$ for t = {t} s')
    else:
        ax.axis('off') 

cbar = fig.colorbar(c, ax=ax_list, label='$Cross velocity$ (mm/s)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(yu), np.max(yu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity - rod in view.svg')
plt.show()



""" Plot 2D mulitple timesteps Vx(x,z) plan XZ before the rod is in the field view """
timesteps = [0, 50, 100, 150] # Time to plot

z0 = 4

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y , V = sample_over_plan(grid, [0,1,0], [0,z0,0], "Velocity")
        Vx = [v[0] for v in V]
        Vy = [v[2] for v in V]
        
        x, y, V = np.array(x), np.array(y), np.array(V)
        xi = np.linspace(x.min(), x.max(), len(xu))
        yi = np.linspace(y.min(), y.max(), len(zu))
        X, Y = np.meshgrid(xi, yi)
        Vx_grid = griddata((x, y), Vx, (X, Y), method='cubic') * 1000
        Vy_grid = griddata((x, y), Vy, (X, Y), method='cubic') * 1000

        step = 1
        c =  ax.pcolormesh(X, Y, Vx_grid, shading='auto', cmap='turbo', vmin=-45, vmax=45)
        ax.quiver(X[::step, ::step], Y[::step, ::step], Vx_grid[::step, ::step], Vy_grid[::step, ::step], color='black', scale=40, scale_units='xy')
        #ax.streamplot(X, Y, Vx_grid, Vy_grid, color='black')
        ax.set_title(f'$Vx$ for t = {t} s')
    else:
        ax.axis('off') 

cbar = fig.colorbar(c, ax=ax_list, label='$Cross velocity$ (mm/s)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(zu), np.max(zu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity XZ- before rod in view.svg')
plt.show()


""" Plot 2D mulitple timesteps Vx(x,z) plan XZ when the rod is in the field view """
timesteps = [160, 210, 330, 380]  # Time to plot
z0 = 4

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y, V = sample_over_plan(grid, [0, 1, 0], [0, z0, 0], "Velocity")
        Vx = [v[0] for v in V]
        Vy = [v[2] for v in V]
        
        x, y, V = np.array(x), np.array(y), np.array(V)
        xi = np.linspace(x.min(), x.max(), len(xu))
        yi = np.linspace(y.min(), y.max(), len(zu))
        X, Y = np.meshgrid(xi, yi)
        Vx_grid = griddata((x, y), Vx, (X, Y), method='cubic')  * 1000
        Vy_grid = griddata((x, y), Vy, (X, Y), method='cubic') * 1000

        step = 1
        c =  ax.pcolormesh(X, Y, Vx_grid, shading='auto', cmap='turbo', vmin=-45, vmax=45)
        ax.quiver(X[::step, ::step], Y[::step, ::step], Vx_grid[::step, ::step], Vy_grid[::step, ::step], color='black', scale=50, scale_units='xy')
        ax.set_title(f'$Vx$ for t = {t} s')
    else:
        ax.axis('off')

cbar = fig.colorbar(c, ax=ax_list, label='$Cross velocity$ (mm/s)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(zu), np.max(zu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity XZ- rod in view.svg')
plt.show()


""" Plot 2D mulitple timesteps Pressure plan XZ when the rod is in the field view """
timesteps = [160, 210, 330, 380, 400, 420, 450, 480, 499]  # Time to plot
z0 = 4

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y, P = sample_over_plan(grid, [0, 1, 0], [0, z0, 0], "Pressure")
        
        x, y, P = np.array(x), np.array(y), np.array(P)
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        P = griddata((x, y), P, (X, Y), method='cubic')  * 1000

        step = 1
        c =  ax.pcolormesh(X, Y, P, shading='auto', cmap='turbo')
        ax.set_title(f'$Pressure$ for t = {t} s')
    else:
        ax.axis('off')

cbar = fig.colorbar(c, ax=ax_list, label='$Pressure$ (Pa)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(zu), np.max(zu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Pressure XZ- rod in view.svg')
plt.show()

""" Plot 2D mulitple timesteps Pressure plan XY when the rod is in the field view """
timesteps = [160, 210, 330, 380, 400, 420, 450, 480, 499]  # Time to plot
z0 = 1

n_rows = int(np.ceil(np.sqrt(len(timesteps))))
fig, axes = plt.subplots(n_rows, n_rows, figsize=(12, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0)
ax_list = axes.flatten()

for i, ax in enumerate(ax_list):
    if i < len(timesteps):
        t = time[timesteps[i]]
        grid = dataset[timesteps[i]]
        x, y, P = sample_over_plan(grid, [0, 0, 1], [0, 0, z0], "Pressure")
        
        x, y, P = np.array(x), np.array(y), np.array(P)
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        P = griddata((x, y), P, (X, Y), method='cubic')  * 1000

        step = 1
        c =  ax.pcolormesh(X, Y, P, shading='auto', cmap='turbo')
        ax.set_title(f'$Pressure$ for t = {t} s')
    else:
        ax.axis('off')

cbar = fig.colorbar(c, ax=ax_list, label='$Pressure$ (Pa)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(yu), np.max(yu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Pressure XZ- rod in view.svg')
plt.show()



