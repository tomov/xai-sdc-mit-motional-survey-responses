import pandas as pd

def excel_column_to_index(column_letter):
    """Convert Excel column letter (e.g., 'A', 'Z', 'AA', 'AG') to 0-based index."""
    index = 0
    for char in column_letter:
        index = index * 26 + (ord(char.upper()) - ord('A') + 1)
    return index - 1

def parse_survey_data(input_file, skip_rows):
    """
    Parse motional_drivers.csv and extract specific columns with new names.
    
    Column mappings:
    - R: participant name
    - AG: CLOSE before
    - AL: CLOSE after
    - AQ: ASV before
    - AV: ASV after
    - BA: BIKE before
    - BF: BIKE after
    """
    # Map Excel columns to 0-based indices
    column_mappings = {
        'R': 'participant name',
        'AG': 'CLOSE before',
        'AL': 'CLOSE after',
        'AQ': 'ASV before',
        'AV': 'ASV after',
        'BA': 'BIKE before',
        'BF': 'BIKE after'
    }
    
    # Convert column letters to indices
    column_indices = {excel_column_to_index(col): new_name 
                      for col, new_name in column_mappings.items()}
    
    # Read CSV, skipping first 3 rows
    df = pd.read_csv(input_file, skiprows=skip_rows)
    
    # Get the indices we need (sorted for order)
    sorted_indices = sorted(column_indices.keys())
    
    # Extract only the columns we need and rename them
    extracted_df = df.iloc[:, sorted_indices].copy()
    extracted_df.columns = [column_indices[idx] for idx in sorted_indices]
    
    return extracted_df

