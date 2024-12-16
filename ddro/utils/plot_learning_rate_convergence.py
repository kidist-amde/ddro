import re
import matplotlib.pyplot as plt
import seaborn as sns

# Apply a modern style
sns.set_theme(style="whitegrid")

# File path to your log file
log_file_path = "logs-sft/url_all.log"

# Lists to store extracted data
epochs = []
learning_rates = []
average_losses = []

# Parse the log file
with open(log_file_path, "r") as file:
    current_epoch = None  # To track the current epoch
    for line in file:
        # Match epoch number
        epoch_match = re.search(r"Epoch (\d+)/\d+", line)
        # Match learning rate
        lr_match = re.search(r"lr: ([\d.e+-]+)", line)
        # Match average loss
        avg_loss_match = re.search(r"Average loss:([\d.e+-]+)", line)

        # Extract epoch if present
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Append data if learning rate is present
        if lr_match:
            epochs.append(current_epoch)
            learning_rates.append(float(lr_match.group(1)))

        # Append data if average loss is present
        if avg_loss_match:
            average_losses.append(float(avg_loss_match.group(1)))

# Ensure all lists are aligned and handle missing data
max_epoch = max(epochs) if epochs else 0
complete_epochs = list(range(1, max_epoch + 1))

# Fill in missing data
final_epochs = []
final_learning_rates = []
final_average_losses = []

for epoch in complete_epochs:
    if epoch in epochs:
        index = epochs.index(epoch)
        final_epochs.append(epoch)
        final_learning_rates.append(learning_rates[index])
        final_average_losses.append(average_losses[index] if index < len(average_losses) else None)
    else:
        # Handle missing data with None or placeholder
        final_epochs.append(epoch)
        final_learning_rates.append(None)  # Placeholder for missing learning rate
        final_average_losses.append(None)  # Placeholder for missing average loss

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot average loss
ax1.plot(final_epochs, final_average_losses, label="Average Loss", color="#1f77b4", linestyle="-", marker="o", markersize=6, linewidth=2)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Average Loss", color="#1f77b4", fontsize=12)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.legend(loc="upper left", fontsize=10)

# Twin axis for learning rate
ax2 = ax1.twinx()
ax2.plot(final_epochs, final_learning_rates, label="Learning Rate", color="#2ca02c", linestyle="--", marker="x", markersize=6, linewidth=2)
ax2.set_ylabel("Learning Rate", color="#2ca02c", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax2.legend(loc="upper right", fontsize=10)

# Title and grid
plt.title("Learning Curve: Average Loss and Learning Rate", fontsize=14, weight='bold')
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

# Save and show the plot
plt.tight_layout()
plt.savefig("SFT_NQ_URL_learning_curve.pdf", format="pdf", dpi=300)  # Save as vector PDF
plt.show()
