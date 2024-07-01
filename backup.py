""" Plot 2D mulitple timesteps Vx(x,y) plan XY for the whole movement """
timesteps = [0, 20, 100, 140, 180, 200, 250, 300, 340, 370, 400, 430, 460] # Time to plot

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

        c =  ax.pcolormesh(X, Y, Vx_grid, shading='auto', cmap='jet', vmin=-45, vmax=45)
        #ax.streamplot(X, Y, Vx_grid, Vy_grid, color='black')
        ax.quiver(X, Y, Vx_grid, Vy_grid, color='black', scale=50, scale_units='xy')
        ax.set_title(f'$Vx$ for t = {t} s')
    else:
        ax.axis('off') 

cbar = fig.colorbar(c, ax=ax_list, label='$Cross velocity$ (mm/s)', orientation='horizontal', pad=0.07)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, 12)
ax.set_ylim(np.min(yu), np.max(yu))
plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Cross velocity XY - whole phases.svg')
plt.show()




""" Plot 2D Cross vorticity for each timesteps (plots saved for video) """

tp = np.arange(0, 499, 1)
PlotData = []
for i in tp:
    grid = dataset[i]
    x, y, V = sample_over_plan(grid, [0, 0, 1], [0, 0, 0.5], "Velocity")
    Vor_x = [v[0] for v in V]
    Vor_y = [v[1] for v in V]
    Vor_z = [v[2] for v in V]
    
    #a, b, V = sample_over_plan(grid, [0, 0, 1], [0, 0, 0.5], "Velocity")
    #Vx = [v[0] for v in V]
    #Vy = [v[1] for v in V]
    
    x, y, V = np.array(x), np.array(y), np.array(V)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    Vor_x = griddata((x, y), Vor_x, (X, Y), method='cubic')
    Vor_y = griddata((x, y), Vor_y, (X, Y), method='cubic')
    Vor_z = griddata((x, y), Vor_z, (X, Y), method='cubic')
    #Vx = griddata((x, y), Vx, (X, Y), method='cubic')
    #Vy = griddata((x, y), Vy, (X, Y), method='cubic') 
    
    Vor_abs = np.sqrt(Vor_x**2 + Vor_y**2 + Vor_z**2)*1000
    fig, ax = plt.subplots()

    c = ax.pcolormesh(X, Y, Vor_abs, cmap='turbo', vmin=0, vmax=45)
    stream = ax.streamplot(X, Y, Vor_x, Vor_y, color='black',)
    cbar = plt.colorbar(c, ax=ax, label='Velocity magnitude [mm/s]')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(0, 12)
    ax.set_ylim(np.min(yu), np.max(yu))
    plt.savefig(f'{SAVE_DIRECTORY}/2D-Plot/Video/{i:05}.tif')
    plt.close()
    
