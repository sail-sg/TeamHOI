import pandas as pd

df = pd.read_csv("log_out.csv", header=None)

#SR
success = df.iloc[:,4]
sr = success.sum() / len(success)
sr = sr

# dist
d = df.iloc[:,5].copy()
d[success == 1] = 0.03
dist = d.mean()

# jerk
jerk = df.iloc[:,6]
jerk_nonzero = jerk[jerk != 0]
jerk_mean = jerk_nonzero.mean()

# t_coop
t_coop = df.iloc[:,7]
lift_t = df.iloc[:,3] - df.iloc[:,2]
lift_valid = lift_t > 10
t_coop_valid = t_coop[lift_valid]
t_coop_mean = t_coop.mean()

# Print metrics
print(f"SR={sr*100:.1f}%, d={dist:.4f},  t_coop={t_coop_mean*100:.1f}%, |J|={jerk_mean:.4f}")

