# %% [markdown]
# # Parse Motional Drivers CSV
# 
# This notebook extracts specific columns from `motional_drivers.csv` and renames them:
# 
# - Column R: participant name
# - Column AG: CLOSE before
# - Column AL: CLOSE after
# - Column AQ: ASV before
# - Column AV: ASV after
# - Column BA: BIKE before
# - Column BF: BIKE after

# %% [code]
from utils import parse_survey_data

# %% [code]
input_file = "data/motional_drivers.csv"
df = parse_survey_data(input_file)
df

# %%
