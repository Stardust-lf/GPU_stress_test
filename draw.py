import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
file_name = "data/INT32_2048.csv"  # Replace with your CSV file name
data = pd.read_csv(file_name)
# Filter data to keep only frequencies between 3000 and 3100 MHz
filtered_data = data

# Initial scatter plot with Frequency (MHz) on x-axis and Error Time (s) on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['Frequency (MHz)'], filtered_data['Error Time (s)'], alpha=0.7, edgecolors='k')

# Adding labels and title
plt.xlabel('Frequency (MHz)', fontsize=12)
plt.ylabel('Error Time (s)', fontsize=12)
# plt.yscale('log')
plt.title('Scatter Plot: Frequency vs Error Time (3000-3100 MHz)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
