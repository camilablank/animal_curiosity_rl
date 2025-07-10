import pandas as pd
import matplotlib.pyplot as plt

# --- Data Simulation (Replace with your actual data loading) ---
# In a real scenario, you would load your data from 'trial_info.csv'
# For example: df = pd.read_csv('trial_info.csv')

# Simulating the 'trial_info' table based on the provided image
csv_path = '/Users/camilablank/Downloads/20240517_117o/trial_info.csv'

df = pd.read_csv(csv_path)

# --- Filter for successful and unsuccessful trials ---
successful_trials_df = df[df['Correct'] == 1].copy()
unsuccessful_trials_df = df[df['Correct'] == -1].copy()

# --- Create the plot ---
plt.figure(figsize=(12, 7)) # Set the figure size for better readability

# Plot successful trials
plt.scatter(
    successful_trials_df['Trial'],
    successful_trials_df['TrialDur'],
    color='skyblue',
    alpha=0.7,
    label='Successful Trials'
)

# Plot unsuccessful trials
plt.scatter(
    unsuccessful_trials_df['Trial'],
    unsuccessful_trials_df['TrialDur'],
    color='salmon',
    alpha=0.7,
    label='Unsuccessful Trials'
)

# Add labels and title
plt.xlabel('Trial Number')
plt.ylabel('Trial Duration (ms)')
plt.title('Trial Number vs. Trial Duration (All Trials)')
plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for easier reading
plt.legend() # Display the legend to differentiate colors

# Display the plot
# In a local environment, this would open a plot window.
# In an online environment, you might need to save it to a file.
plt.show()
