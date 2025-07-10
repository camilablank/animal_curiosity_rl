import pandas as pd
import matplotlib.pyplot as plt

csv_path = '/Users/camilablank/Downloads/20240517_117o/trial_info.csv'

df = pd.read_csv(csv_path)

successful_trials_df = df[df['Correct'] == 1].copy()
unsuccessful_trials_df = df[df['Correct'] == -1].copy()

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
plt.show()
