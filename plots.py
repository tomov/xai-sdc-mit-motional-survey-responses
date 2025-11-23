# %% [code]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# List of CSV files to analyze (can be extended with more files)
csv_files = [
    'results/motional_drivers_results_FINAL.csv',
    'results/normal_people_results_FINAL.csv',
]

scenario_columns = ['CLOSE', 'ASV', 'BIKE']
anchor_labels = ['Initial Belief', 'Final Belief', 'Ground Truth']
candidate_labels = ['Candidate 1', 'Candidate 2']

# Create figure with n_rows (one per dataset) and 3 columns (one per scenario)
n_rows = len(csv_files)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

# Handle case where there's only one row (axes becomes 1D)
if n_rows == 1:
    axes = axes.reshape(1, -1)

for row_idx, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    
    # Extract dataset name from filename for labeling
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    for col_idx, scenario in enumerate(scenario_columns):
        confusion_matrix = np.zeros((2, 3), dtype=int)
        
        for _, row in df.iterrows():
            responses = row[scenario].strip().split()
            
            for anchor_idx, response in enumerate(responses):
                candidate_idx = int(response) - 1
                confusion_matrix[candidate_idx, anchor_idx] += 1
        
        ax = axes[row_idx, col_idx]
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(anchor_labels)
        ax.set_yticklabels(candidate_labels)
        
        for i in range(2):
            for j in range(3):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')
        
        # Add scenario title for top row, dataset name for first column
        if row_idx == 0:
            ax.set_title(f'{scenario} Scenario', fontsize=14, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f'{dataset_name}\nCandidate', fontsize=12)
        
        ax.set_xlabel('Anchor', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Count')

# %% [code]

plt.tight_layout()
fig.savefig('results/confusion_matrices.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
