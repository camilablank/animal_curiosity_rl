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
    df = pd.read_csv('/Users/camilablank/Downloads/20240517_117o/nose.csv', header=[0])
except FileNotFoundError:
    print("Error: Trajectory CSV file not found. Please check the path and filename.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Access nose data.
try:
    nose_x_raw = df['x'].values # Access by column name 'x'
    nose_y_raw = df['y'].values # Access by column name 'y'
    print(f"Extracted nose X from column: 'x'")
    print(f"Extracted nose Y from column: 'y'")
except KeyError as e:
    print(f"Error: Expected column '{e}' not found in CSV. Please check column names.")
    exit()

nose_x_raw = pd.to_numeric(nose_x_raw, errors='coerce')
nose_y_raw = pd.to_numeric(nose_y_raw, errors='coerce')

# Remove any NaN values from the nose data (these might arise from 'coerce' above)
valid_indices = ~np.isnan(nose_x_raw) & ~np.isnan(nose_y_raw)
nose_x_raw_clean = nose_x_raw[valid_indices]
nose_y_raw_clean = nose_y_raw[valid_indices]

if len(nose_x_raw_clean) == 0:
    print("No valid nose coordinate data found after cleaning. Cannot plot trajectory.")
    exit()

# Create the maze object
maze_size = 6 # Set to an even number, typically 6 for this maze type
maze = NewMaze(maze_size)

# Determine maze plot limits and scale
min_maze_x = np.min(maze.xc)
max_maze_x = np.max(maze.xc)
min_maze_y = np.min(maze.yc)
max_maze_y = np.max(maze.yc)

# Determine the range of raw trajectory data
raw_x_min = np.min(nose_x_raw)
raw_x_max = np.max(nose_x_raw)
raw_y_min = np.min(nose_y_raw)
raw_y_max = np.max(nose_y_raw)

# Calculate the scaling factor and offset to map raw data to maze coordinates
epsilon = 1e-9

# Calculate scaling factors to fit trajectory into maze
scale_x = (max_maze_x - min_maze_x) / (raw_x_max - raw_x_min + epsilon)
scale_y = (max_maze_y - min_maze_y) / (raw_y_max - raw_y_min + epsilon)

# Calculate offsets to center trajectory in maze
offset_x = min_maze_x + (max_maze_x - min_maze_x) / 2 - (raw_x_min + (raw_x_max - raw_x_min) / 2) * scale_x
offset_y = min_maze_y + (max_maze_y - min_maze_y) / 2 - (raw_y_min + (raw_y_max - raw_y_min) / 2) * scale_y

# Apply scaling and offset
nose_x_scaled = nose_x_raw * scale_x + offset_x
nose_y_scaled = nose_y_raw * scale_y + offset_y

# Expand trajectory vertically and horizontally
vertical_expansion = 1.1
traj_center_y = np.mean(nose_y_scaled)
nose_y_scaled = (nose_y_scaled - traj_center_y) * vertical_expansion + traj_center_y

horizontal_expansion = 1.05
traj_center_x = np.mean(nose_y_scaled)
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

# Add a colorbar to explain the color mapping
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


