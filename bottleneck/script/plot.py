import matplotlib.pyplot as plt
plt.ion()
# Data for Problem Radius and accuracies
problem_radius = list(range(2, 9))  # Problem Radius from 2 to 8
accuracy_gat = [100, 100, 100, 100, 60.8, 57.2, 54]  # Accuracy for GAT
accuracy_gin = [100, 100, 100, 80, 18, 11, 5]  # Accuracy for GIN
accuracy_gcn = [100, 100, 100, 58, 4, 3, 3]  # Accuracy for GCN
accuracy_ggnn = [100, 100, 100, 20, 4, 3, 3]  # Accuracy for GGNN
accuracy_fswgnn = [100] * 7  # Accuracy for FSW-GNN (Ours)

# Create the plot
plt.figure(figsize=(8, 6))


# Plotting each model
plt.plot(problem_radius, accuracy_gat, marker='o', linestyle='-', linewidth=3, label='GAT')
plt.plot(problem_radius, accuracy_gin, marker='D', linestyle='-', linewidth=3, label='GIN')
plt.plot(problem_radius, accuracy_gcn, marker='v', linestyle='-', linewidth=3, label='GCN')
plt.plot(problem_radius, accuracy_ggnn, marker='s', linestyle='-', linewidth=3, label='GGNN')
plt.plot(problem_radius, accuracy_fswgnn, marker='^', linestyle='-', linewidth=3, label='FSW-GNN (Ours)')

# Adding labels, title, and legend
plt.xlabel('Problem Radius')
plt.ylabel('Accuracy (%)')
plt.title('Tree')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=16)

# Display the plot
plt.show()

