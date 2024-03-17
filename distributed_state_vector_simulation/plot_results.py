import pandas as pd
import matplotlib.pyplot as plt
import sys

# Convert data to dataframe
df = pd.read_csv(sys.argv[1])

# Create stacked bar plot
plt.figure(figsize=(10, 6))
plt.bar(df['Qubits'], df['pilotStartupTime(Sec)'], label='pilotStartupTime', color='skyblue')
plt.bar(df['Qubits'], df['runTime (Sec)'], bottom=df['pilotStartupTime(Sec)'], label='runTime', color='orange')

plt.xlabel('Qubits')
plt.ylabel('Total Run Time (Sec)')
plt.title('Total Run Time by Qubits')
plt.legend()
plt.xticks(df['Qubits'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

