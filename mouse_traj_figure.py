import os
import sys

script_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap module
from matplotlib.colors import Normalize
from matplotlib import patches
from dataclasses import make_dataclass
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from plot_utils.MM_plot_util import plot, set_axes
from plot_utils.MM_maze_util import Maze, NewMaze, PlotMazeWall, PlotMazeFunction
# from plot_utils.color_map import generate_colormap

def generate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ truncates a colormap 
        args:
            cmap (object): a colormap
            min_val (float): min val at which to truncate
            max_val (float): max val at which to truncate
            n (int): number of divisions of the colormap to decie truncation points 
        returns:
            new_cmap (object): new truncated cmap """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Load coordinate data from csv file
try:
    df = pd.read_csv('/Users/camilablank/Downloads/kara_meister_maze_data/20240507_117LDLC_resnet50_MazeJul20shuffle1_2060000_filtered.csv', header=[0, 1, 2])
except FileNotFoundError:
    print("Error: Trajectory CSV file not found. Please check the path and filename.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Access nose data
nose_cols = [col for col in df.columns if col[1] == 'nose' and col[2] in ['x', 'y']] # Be explicit with 'x' and 'y'

if len(nose_cols) >= 2:
    # Assuming the order found by list comprehension is (..., 'x') then (..., 'y')
    # If not, you might need to explicitly pick based on col[2]
    nose_x_col_name = None
    nose_y_col_name = None
    for col_tuple in nose_cols:
        if col_tuple[2] == 'x':
            nose_x_col_name = col_tuple
        elif col_tuple[2] == 'y':
            nose_y_col_name = col_tuple
    
    if nose_x_col_name and nose_y_col_name:
        nose_x_raw = df[nose_x_col_name].values
        nose_y_raw = df[nose_y_col_name].values
        
        # Convert to numeric values and handle any non-numeric data
        nose_x_raw = pd.to_numeric(nose_x_raw, errors='coerce')
        nose_y_raw = pd.to_numeric(nose_y_raw, errors='coerce')
        
        print(f"Extracted nose X from column: {nose_x_col_name}")
        print(f"Extracted nose Y from column: {nose_y_col_name}")
    else:
        print("Could not find both 'x' and 'y' under 'nose' bodypart. Check column names.")
        exit()
else:
    print("Could not find at least two 'nose' columns under any multi-index header (x and y).")
    print("Please inspect your CSV file's exact column structure manually.")
    exit()

# Remove any NaN values from the nose data
valid_indices = ~np.isnan(nose_x_raw) & ~np.isnan(nose_y_raw)
nose_x_raw_clean = nose_x_raw[valid_indices]
nose_y_raw_clean = nose_y_raw[valid_indices]

if len(nose_x_raw_clean) == 0:
    print("No valid nose coordinate data found after cleaning. Cannot plot trajectory.")
    exit()

# Create maze object
maze_size = 6 # Set to an even number, typically 6 for this maze type
maze = NewMaze(maze_size)

# Determine maze plot limits
min_maze_x = np.min(maze.xc)
max_maze_x = np.max(maze.xc)
min_maze_y = np.min(maze.yc)
max_maze_y = np.max(maze.yc)

# Determine the range of raw trajectory data
raw_x_min = np.min(nose_x_raw_clean)
raw_x_max = np.max(nose_x_raw_clean)
raw_y_min = np.min(nose_y_raw_clean)
raw_y_max = np.max(nose_y_raw_clean)

# Adjust the raw data range to make trajectory wider horizontally
x_range = raw_x_max - raw_x_min
x_center = (raw_x_max + raw_x_min) / 2
raw_x_min = x_center - x_range * 0.6
raw_x_max = x_center + x_range * 0.6

epsilon = 1e-9

scale_x = (max_maze_x - min_maze_x) / (raw_x_max - raw_x_min + epsilon)
scale_y = (max_maze_y - min_maze_y) / (raw_y_max - raw_y_min + epsilon)

offset_x = min_maze_x + (max_maze_x - min_maze_x) / 2 - (raw_x_min + (raw_x_max - raw_x_min) / 2) * scale_x
offset_y = min_maze_y + (max_maze_y - min_maze_y) / 2 - (raw_y_min + (raw_y_max - raw_y_min) / 2) * scale_y

# Apply scaling and offset
nose_x_scaled = nose_x_raw_clean * scale_x + offset_x
nose_y_scaled = nose_y_raw_clean * scale_y + offset_y

# Expand trajectory vertically
vertical_expansion = 1.15  # Make it 15% taller
traj_center_y = np.mean(nose_y_scaled)
nose_y_scaled = (nose_y_scaled - traj_center_y) * vertical_expansion + traj_center_y

# Expand trajectory horizontally
horizontal_expansion = 1.08  # Make it 8% wider
traj_center_x = np.mean(nose_x_scaled)
nose_x_scaled = (nose_x_scaled - traj_center_x) * horizontal_expansion + traj_center_x

print(f"\nRaw X range: [{raw_x_min:.2f}, {raw_x_max:.2f}]")
print(f"Raw Y range: [{raw_y_min:.2f}, {raw_y_max:.2f}]")
print(f"Maze X range: [{min_maze_x:.2f}, {max_maze_x:.2f}]")
print(f"Maze Y range: [{min_maze_y:.2f}, {max_maze_y:.2f}]")
print(f"Applied X scale: {scale_x:.2f}, offset: {offset_x:.2f}")
print(f"Applied Y scale: {scale_y:.2f}, offset: {offset_y:.2f}")
print(f"Scaled X range: [{np.min(nose_x_scaled):.2f}, {np.max(nose_x_scaled):.2f}]")
print(f"Scaled Y range: [{np.min(nose_y_scaled):.2f}, {np.max(nose_y_scaled):.2f}]")

# Rotate trajectory
temp_x = nose_x_scaled.copy()
temp_y = nose_y_scaled.copy()
nose_x_scaled = max_maze_y - temp_y + min_maze_y
nose_y_scaled = temp_x

# Set plot limits with some padding around the maze
x_plot_lim = [min_maze_x - 1, max_maze_x + 1]
y_plot_lim = [min_maze_y - 1, max_maze_y + 1]

# Create the plot figure and axes
fig_size = 8
fig, ax = plt.subplots(figsize=(fig_size, fig_size))

maze_axes = PlotMazeWall(maze, axes=ax, figsize=fig_size)

points = np.array([nose_x_scaled, nose_y_scaled]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

time_for_coloring = np.arange(len(nose_x_scaled))
base_cmap = cm.get_cmap('viridis') # Use regular viridis instead of viridis_r

display_cmap = generate_colormap(base_cmap, minval=0.0, maxval=1.0) # Using full range for now

norm = Normalize(vmin=time_for_coloring.min(), vmax=time_for_coloring.max())

# Create the LineCollection
lc = LineCollection(segments.tolist(), cmap=display_cmap, norm=norm, linewidth=2)
lc.set_array(time_for_coloring) # Assign the normalized time values to color the segments

# Add the LineCollection to the maze axes
line = maze_axes.add_collection(lc)

# Add a colorbar
cbar_ax = fig.add_axes((0.92, 0.25, 0.02, 0.5)) # Use tuple instead of list
cbar = fig.colorbar(line, cax=cbar_ax)
cbar.set_label('Time in Trajectory', rotation=270, labelpad=15) # Label the colorbar
cbar.set_ticks([time_for_coloring.min(), time_for_coloring.max()])
cbar.set_ticklabels(['Entry', 'Exit']) # Label the ends of the colorbar

# Ensure plot limits are set by the maze
set_axes(maze_axes,
         xlabel='X Position (Scaled Maze Units)',
         ylabel='Y Position (Scaled Maze Units)',
         legend=[],
         loc=None,  
         xlim=x_plot_lim,
         ylim=y_plot_lim,
         xscale='linear', 
         yscale='linear', 
         xticks=None, 
         yticks=None, 
         xhide=True,
         yhide=True,
         yrot=False, 
         yzero=False, 
         yflip=True,
         grid=False, 
         equal=True
        )

# Mark start and end points of the trajectory with specific colors
if len(nose_x_scaled) > 0:
    # Get the specific colors for the start and end points from the colormap
    start_color = 'red'
    end_color = 'red'

    maze_axes.plot(nose_x_scaled[0], nose_y_scaled[0], 'o', markersize=10,
                   color=start_color, label='Start', zorder=5)
    maze_axes.plot(nose_x_scaled[-1], nose_y_scaled[-1], 'x', markersize=10,
                   color=end_color, label='End', zorder=5)
    maze_axes.legend(loc='upper right')

plt.tight_layout(rect=(0, 0, 0.9, 1)) # Use tuple instead of list
plt.show()


