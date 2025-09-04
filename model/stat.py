import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#sensor_id,Latitude,Longitude,lanes,maxspeed,ref,direction,Time,AggSpeed,% Observed,weather
def pre_train_stat():
        data_path = "/Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/data_process/exmaple.csv"

        df = pd.read_csv(data_path)

        high_speed_count = df[df["AggSpeed"] > 55].shape[0]

        medium_speed_count = df[(df["AggSpeed"] > 35) & (df["AggSpeed"] <= 55)].shape[0]

        low_speed_count = df[df["AggSpeed"] <= 35].shape[0]

        print(high_speed_count)
        print(medium_speed_count)
        print(low_speed_count)


        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["hour"] = df["Time"].dt.hour

        # Categorize speeds
        df["speed_category"] = pd.cut(
            df["AggSpeed"],
            bins=[-float("inf"), 35, 55, float("inf")],
            labels=["Low Speed", "Medium Speed", "High Speed"]
        )

        # Count per hour per category
        counts = df.groupby(["hour", "speed_category"]).size().unstack(fill_value=0)

        # Plot (grouped bars per hour)
        counts.plot(kind="bar", figsize=(12, 6))
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Count")
        plt.title("Number of Low / Medium / High Speeds per Hour")
        plt.xticks(rotation=0)
        plt.legend(title="Speed Category")
        plt.tight_layout()
        plt.show()

def post_train_stat():
    # path to your predictions file
    csv_path = "/Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/test_preds_balanced.csv"

    df = pd.read_csv(csv_path)

    # define bins
    bins = {
        "low (≤35)": df["y_true"] <= 35,
        "medium (35–55]": (df["y_true"] > 35) & (df["y_true"] <= 55),
        "high (>55)": df["y_true"] > 55,
    }

    # compute MAE per bin
    for name, mask in bins.items():
        subset = df.loc[mask]
        if len(subset) == 0:
            print(f"{name}: no samples")
            continue
        mae = np.mean(np.abs(subset["y_pred"] - subset["y_true"]))
        print(f"{name}: MAE = {mae:.3f} over {len(subset)} samples")

    # overall MAE as well
    overall_mae = np.mean(np.abs(df["y_pred"] - df["y_true"]))
    print(f"\nOverall MAE = {overall_mae:.3f} over {len(df)} samples")


if __name__ == "__main__":
    #pre_train_stat()
    post_train_stat()