import pandas as pd

def read_ts_file(filepath, verbose=True):
    data_started = False
    data_content = "" # Use a string to accumulate the single long data line
    metadata = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif line.startswith('@data'):
                data_started = True
                continue # Move to the next line to read the data
            elif line.startswith('@'):
                parts = line[1:].split(' ', 1)
                key = parts[0]
                value = parts[1] if len(parts) == 2 else 'True'
                
                if value.lower() == 'true':
                    metadata[key] = True
                elif value.lower() == 'false':
                    metadata[key] = False
                elif value.isdigit():
                    metadata[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    metadata[key] = float(value)
                else:
                    metadata[key] = value
            elif data_started:
                # Accumulate the data content (assuming it's all on one line or contiguous)
                data_content += line
                # Break after reading the first data line if we expect only one long series
                # If your file has multiple such long lines for different series, remove this break
                break 

    if not data_content:
        return pd.DataFrame(), metadata

    all_series_data = []
    
    # The entire series string needs to be split by the pattern '),('
    # Example: (ts,val),(ts,val)
    # The first pair will start with '(', the last will end with ')'
    
    # Step 1: Remove the outer parentheses from the entire data string
    cleaned_data_content = data_content.strip('()')

    # Step 2: Split by the common separator between pairs
    # This will give us strings like "2016-03-25 17:00:00,22.6"
    individual_pairs = cleaned_data_content.split('),(')

    for pair_str in individual_pairs:
        parts = pair_str.split(',', 1) # Split only on the first comma
        if len(parts) == 2:
            ts_str, val_str = parts[0].strip("'\""), parts[1]
            try:
                all_series_data.append({'timestamp': pd.to_datetime(ts_str), 'value': float(val_str)})
            except ValueError:
                if verbose:
                    print(f"Warning: Could not parse data point '{pair_str}' due to format issues.")
                pass
        else:
            if verbose:
                print(f"Warning: Malformed data block found: '{pair_str}'. Expected 'timestamp,value'.")

    df = pd.DataFrame(all_series_data)
    
    if metadata.get('timestamps') and 'timestamp' in df.columns:
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

    return df, metadata

# Usage with your specified file path
file_path = "/Users/fouadabiad/Downloads/3902637/AppliancesEnergy_TEST.ts"
try:
    # Set verbose=False to suppress the warnings
    df, metadata = read_ts_file(file_path, verbose=False) 

    print("--- DataFrame Head ---")
    print(df.head())
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Metadata ---")
    print(metadata)
except KeyboardInterrupt:
    print("\nProcess interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn error occurred: {e}")