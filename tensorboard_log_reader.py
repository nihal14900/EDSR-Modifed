import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to your TensorBoard log directory
# log_dir = r"C:\Users\nihal\Desktop\Final\EDSR-PyTorch\samples\logs\model18-div2kmean"
log_dir = r"C:\Users\nihal\Desktop\Final\EDSR-PyTorch\samples\logs\model15-df2k"


# Find all event files in the directory
event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]

if not event_files:
    raise FileNotFoundError("No TensorBoard event files found in the directory!")

# Dictionary to store aggregated log data
combined_log_data = {}

# Load and combine data from multiple event files
for event_file in event_files:
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Get available scalar tags
    scalar_tags = event_acc.Tags()["scalars"]

    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        if tag not in combined_log_data:
            combined_log_data[tag] = {}

        # Store values for each step
        for event in events:
            step = event.step
            value = event.value

            if step not in combined_log_data[tag]:
                combined_log_data[tag][step] = []

            combined_log_data[tag][step].append(value)  # Store values for averaging later

# Convert collected values into lists and compute statistics
final_log_data = {}
for tag, steps_dict in combined_log_data.items():
    sorted_steps = sorted(steps_dict.keys())  # Ensure step order
    aggregated_values = [np.mean(steps_dict[step]) for step in sorted_steps]  # Compute average value at each step

    final_log_data[tag] = list(zip(sorted_steps, aggregated_values))  # Store as step-value pairs

# Determine number of rows and columns for subplots
num_plots = len(final_log_data)
cols = min(3, num_plots)  # Max 3 columns for better readability
rows = math.ceil(num_plots / cols)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()  # Flatten in case of multiple rows/columns

# Use Matplotlib's color cycle for distinct colors
colors = plt.cm.tab10.colors

# Plot each scalar with min, max, avg values
for ax, (tag, data), color in zip(axes, final_log_data.items(), colors):
    steps, values = zip(*data)  # Unpack step-value pairs
    values = np.array(values)  # Convert to NumPy array for stats

    # Compute min, max, avg
    min_val, max_val, avg_val = values.min(), values.max(), values.mean()

    # Plot the data
    ax.plot(steps, values, marker="", linestyle="-", color=color, label=tag)

    # Annotate min, max, avg
    ax.axhline(min_val, color="red", linestyle="--", alpha=0.5, label=f"Min: {min_val:.4f}")
    ax.axhline(max_val, color="green", linestyle="--", alpha=0.5, label=f"Max: {max_val:.4f}")
    ax.axhline(avg_val, color="blue", linestyle="--", alpha=0.5, label=f"Avg: {avg_val:.4f}")

    # Add text for min, max, avg
    ax.text(steps[-1], min_val, f"Min: {min_val:.4f}", color="red", fontsize=10, verticalalignment='bottom')
    ax.text(steps[-1], max_val, f"Max: {max_val:.4f}", color="green", fontsize=10, verticalalignment='top')
    ax.text(steps[-1], avg_val, f"Avg: {avg_val:.4f}", color="blue", fontsize=10, verticalalignment='center')

    # Formatting
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(f"{tag} (Avg across runs)")
    ax.legend()
    ax.grid(True)

# Remove empty subplots (if any)
for i in range(len(final_log_data), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout and show all plots
plt.tight_layout()
plt.show()
