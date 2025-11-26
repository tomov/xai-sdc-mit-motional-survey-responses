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

# Mapping from CSV files to their proper titles
csv_titles = {
    'results/motional_drivers_results_FINAL.csv': 'Experts',
    'results/normal_people_results_FINAL.csv': 'Non-experts',
}

scenario_columns = ['CLOSE', 'ASV', 'BIKE']
anchor_labels = ['Before explanation', 'After explanation', 'Ground truth']
candidate_labels = ['Before explanation', 'After explanation']

# Create separate figure for each dataset (1 row, 4 columns each)
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # Extract dataset name from filename for labeling
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    # Get proper title for the figure
    figure_title = csv_titles.get(csv_file, dataset_name)
    
    # Create figure with 1 row and 4 columns (aggregated + 3 scenarios)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    
    # First, compute aggregated confusion matrix across all scenarios
    aggregated_matrix = np.zeros((2, 3), dtype=int)
    
    for _, row in df.iterrows():
        for scenario in scenario_columns:
            responses = row[scenario].strip().split()
            
            for anchor_idx, response in enumerate(responses):
                candidate_idx = int(response) - 1
                aggregated_matrix[candidate_idx, anchor_idx] += 1
    
    # Plot aggregated matrix in column 0
    ax = axes[0]
    im = ax.imshow(aggregated_matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(anchor_labels, rotation=30)
    ax.set_yticklabels(candidate_labels, rotation=70)
    
    for i in range(2):
        for j in range(3):
            text = ax.text(j, i, aggregated_matrix[i, j],
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title('All scenarios', fontsize=14, fontweight='bold')
    ax.set_xlabel('On-road safety driver belief', fontsize=12)
    ax.set_ylabel('Online study\nparticipant belief', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Count')
    
    # Now plot individual scenarios in columns 1, 2, 3
    for col_idx, scenario in enumerate(scenario_columns):
        confusion_matrix = np.zeros((2, 3), dtype=int)
        
        for _, row in df.iterrows():
            responses = row[scenario].strip().split()
            
            for anchor_idx, response in enumerate(responses):
                candidate_idx = int(response) - 1
                confusion_matrix[candidate_idx, anchor_idx] += 1
        
        ax = axes[col_idx + 1]
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(anchor_labels, rotation=30)
        ax.set_yticklabels(candidate_labels, rotation=70)
        
        for i in range(2):
            for j in range(3):
                text = ax.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')
        
        ax.set_title(f'{scenario} scenario', fontsize=14, fontweight='bold')
        ax.set_xlabel('On-road safety driver belief', fontsize=12)
       
        plt.colorbar(im, ax=ax, label='Count')
    
    fig.suptitle(figure_title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'results/confusion_matrices_{dataset_name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# %%

# %%
