# %% [code]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results/normal_people_results_FINAL.csv')

scenario_columns = ['CLOSE', 'ASV', 'BIKE']
anchor_labels = ['Initial Belief', 'Final Belief', 'Ground Truth']
candidate_labels = ['Candidate 1', 'Candidate 2']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, scenario in enumerate(scenario_columns):
    confusion_matrix = np.zeros((2, 3), dtype=int)
    
    for _, row in df.iterrows():
        responses = row[scenario].strip().split()
        
        for anchor_idx, response in enumerate(responses):
            candidate_idx = int(response) - 1
            confusion_matrix[candidate_idx, anchor_idx] += 1
    
    ax = axes[idx]
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(anchor_labels)
    ax.set_yticklabels(candidate_labels)
    
    for i in range(2):
        for j in range(3):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title(f'{scenario} Scenario', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Anchor', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Candidate', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Count')

# %% [code]

plt.tight_layout()
plt.show()


# %% [code]
plt.savefig('results/confusion_matrices.pdf', dpi=300, bbox_inches='tight')
