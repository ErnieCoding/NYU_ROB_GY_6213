import pandas as pd
import glob
import os

directory_path = "./data"
output_file = "combined_data.csv"

all_files = glob.glob(os.path.join(directory_path, "*.csv"))

df_list = [pd.read_csv(f) for f in all_files]

combined_df = pd.concat(df_list, axis=0, ignore_index=True)

combined_df.to_csv(output_file, index=False)

print(f"Successfully merged {len(all_files)} files into {output_file}")

