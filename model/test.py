import pandas as pd

src = "/Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/data_process/exmaple.csv"
out = "/Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/data_process/example.csv"

df = pd.read_csv(src)
df["sensor_id"] = (
    df["Latitude"].round(6).astype(str) + ";" + df["Longitude"].round(6).astype(str)
)
df = df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)

print("unique sensors before:", pd.read_csv(src)["sensor_id"].nunique())
print("unique sensors after :", df["sensor_id"].nunique())

df.to_csv(out, index=False)
print("Wrote:", out)
