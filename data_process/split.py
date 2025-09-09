import pandas as pd

allowed = ["CA 134", "CA 170","I 605"]

df = pd.read_csv("/Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/old_data/split_df_v2.csv")

df = df[df["road_name"].isin(allowed)]

df.to_csv("weights_test.csv", index=False)