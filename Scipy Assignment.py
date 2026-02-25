import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
from scipy.interpolate import interp1d

df = pd.read_csv("players_stats_by_season_full_details.csv")

regular_df = df[df["Stage"] == "Regular_Season"].copy()

season_counts = regular_df.groupby("Player")["Season"].nunique()
top_player = season_counts.idxmax()
print(f"Player with most regular seasons: {top_player}")

player_df = regular_df[regular_df["Player"] == top_player].copy()

player_df["season_year"] = player_df["Season"].str[:4].astype(int)
player_df = player_df.sort_values("season_year")

player_df["three_pt_accuracy"] = player_df["3PM"] / player_df["3PA"]

reg_data = player_df.dropna(subset=["three_pt_accuracy"])

years = reg_data["season_year"].values
accuracy = reg_data["three_pt_accuracy"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(years, accuracy)

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")

def fit_function(x):
    return slope * x + intercept

integral, _ = integrate.quad(fit_function, years.min(), years.max())
average_integrated = integral / (years.max() - years.min())
actual_average = np.mean(accuracy)

print(f"Integrated Average Accuracy: {average_integrated}")
print(f"Actual Average Accuracy: {actual_average}")

interp_func = interp1d(years, accuracy, kind='linear', fill_value="extrapolate")

missing_years = np.array([2002, 2015])
estimated_values = interp_func(missing_years)

print(f"Estimated 2002-03 Accuracy: {estimated_values[0]}")
print(f"Estimated 2015-16 Accuracy: {estimated_values[1]}")

FGM = regular_df["FGM"].dropna().values
FGA = regular_df["FGA"].dropna().values

print("\n--- Whole Dataset (Regular Season) Statistics ---")
print(f"FGM Mean: {np.mean(FGM):.2f}, Var: {np.var(FGM):.2f}, Skew: {stats.skew(FGM):.2f}")
print(f"FGA Mean: {np.mean(FGA):.2f}, Var: {np.var(FGA):.2f}, Skew: {stats.skew(FGA):.2f}")

paired_t = stats.ttest_rel(FGM, FGA)
print(f"Paired t-test: {paired_t}")

independent_t = stats.ttest_ind(FGM, FGA)
print(f"Independent t-test: {independent_t}")