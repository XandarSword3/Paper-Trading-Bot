import pandas as pd

trials_path = "quant_research/trials.csv"
df = pd.read_csv(trials_path)
initial_len = len(df)
df_deduped = df.drop_duplicates(subset=["trial_id"], keep="last").reset_index(drop=True)
final_len = len(df_deduped)

df_deduped.to_csv(trials_path, index=False)
print(f"Deduplicated {trials_path}: {initial_len} -> {final_len} rows ({initial_len - final_len} duplicates removed).")
