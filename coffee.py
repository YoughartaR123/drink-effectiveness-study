# ================================================
# Project: drink-effectiveness-study
# Author: Yougharta
# ================================================

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
from colorama import Fore, Style






# ===============================
# 1. Utility Functions
# ===============================

def ks_with_sample_params(x):
    """K-S against normal with sample mean/std (use ddof=1 for std)."""
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    stat, p = stats.kstest(x, 'norm', args=(mu, sigma))
    return p


def cohens_d_paired(x_before, x_after):
    """Cohen's d for paired samples."""
    d = x_after - x_before
    mean_diff = np.mean(d)
    sd_diff = np.std(d, ddof=1)
    if sd_diff == 0:
        return np.nan
    return mean_diff / sd_diff


def cliffs_delta(x, y):
    """Cliff’s delta for non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    gt = lt = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                gt += 1
            elif xi < yj:
                lt += 1
    return (gt - lt) / (n_x * n_y)


def find_most_effective_group(effects: dict):
    """Find the group with the strongest overall effect."""
    abs_effects = {pair: abs(val) for pair, val in effects.items()}
    groups = {}
    for (g1, g2), val in abs_effects.items():
        groups.setdefault(g1, []).append(val)
        groups.setdefault(g2, []).append(val)
    avg_magnitude = {g: np.mean(vals) for g, vals in groups.items()}
    most_effective = max(avg_magnitude, key=avg_magnitude.get)
    return avg_magnitude, most_effective


# ===============================
# 2. Load & Prepare Data
# ===============================


# Simulate data (reproducible)


np.random.seed(42)
n_participants = 20
participants = np.arange(1, n_participants + 1)

# baseline productivity for each participant (tasks/hour)
baseline = np.random.normal(loc=50, scale=5, size=n_participants)

# true (population) effects by drink (small differences)
effects = {
    'Regular':  1.5,   # +1.5 tasks
    'Decaf':   -0.5,   # -0.5 tasks
    'Energy':   3.0    # +3.0 tasks
}

# within-subject noise (measurement variability)
within_sd = 3.0

# create long-format dataframe
rows = []
for i, subj in enumerate(participants):
    for drink, eff in effects.items():
        val = baseline[i] + eff + np.random.normal(0, within_sd)
        rows.append({'participant': subj, 'drink': drink, 'tasks': val})

df = pd.DataFrame(rows)

# inject outliers into ~7% of observations
n_obs = len(df)
n_outliers = max(1, int(n_obs * 0.07))
outlier_idx = np.random.choice(df.index, size=n_outliers, replace=False)
# randomly choose some outliers high, some low
for idx in outlier_idx:
    df.at[idx, 'tasks'] += np.random.choice([20, -15])  # big jump

# shuffle rows so order is not grouped
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

#separate each group
group_A = df[df['drink'] == 'Regular']['tasks'].values
group_B = df[df['drink'] == 'Decaf']['tasks'].values
group_C = df[df['drink'] == 'Energy']['tasks'].values


# ===============================
# 3. Visualizations (EDA)
# ===============================

plt.figure(figsize=(10, 5))
sns.boxplot(x="drink", y="tasks", data=df, palette="Set2")
sns.swarmplot(x="drink", y="tasks", data=df, color=".25")
plt.title("Distribution of Tasks Completed by Drink Type")
plt.show()

plt.figure(figsize=(10, 5))
sns.violinplot(x="drink", y="tasks", data=df, palette="Set3", inner="quartile")
plt.title("Violin Plot: Task Distribution per Drink")
plt.show()



# ===============================
# 4. Assumption Checks
# ===============================

print(Fore.CYAN + Style.BRIGHT + "\n=== Normality Check (paired differences) ===" + Style.RESET_ALL)
diffs = {
    "A-B": group_A - group_B,
    "A-C": group_A - group_C,
    "B-C": group_B - group_C
}
for name, d in diffs.items():
    p_shapiro = stats.shapiro(d)[1]
    print(f"{name}: Shapiro-Wilk p = {p_shapiro:.4f}")

print(Fore.CYAN + Style.BRIGHT + "\n=== Variance (Sphericity Proxy) ===" + Style.RESET_ALL)
# Note: For real repeated-measures ANOVA, use Mauchly’s test (statsmodels).
levene_p = stats.levene(group_A, group_B, group_C)[1]
print(f"Levene’s p = {levene_p:.4f}")

alpha = 0.05
all_normal = all(stats.shapiro(d)[1] > alpha for d in diffs.values())
equal_var = levene_p > alpha

# ===============================
# 5. Hypothesis Testing
# ===============================

if all_normal and equal_var:
    test_name = "Repeated-measures ANOVA"
    print(Fore.CYAN + Style.BRIGHT + f"\n=== Using {test_name} ===" + Style.RESET_ALL)
    stat, p_value = stats.f_oneway(group_A, group_B, group_C)
else:
    test_name = "Friedman Test"
    print(Fore.CYAN + Style.BRIGHT + f"\n=== Using {test_name} ===" + Style.RESET_ALL)
    stat, p_value = stats.friedmanchisquare(group_A, group_B, group_C)

print(f"Statistic = {stat:.4f}, p = {p_value:.4f}")

# ===============================
# 6. Post-hoc Analysis & Effect Size
# ===============================

if p_value < alpha:
    print(Fore.RED + Style.BRIGHT + "\nSignificant differences detected!" + Style.RESET_ALL)

    if test_name == "Repeated-measures ANOVA":
        effects = {
            ('A', 'B'): cohens_d_paired(group_A, group_B),
            ('A', 'C'): cohens_d_paired(group_A, group_C),
            ('B', 'C'): cohens_d_paired(group_B, group_C)
        }
    else:
        effects = {
            ('A', 'B'): cliffs_delta(group_A, group_B),
            ('A', 'C'): cliffs_delta(group_A, group_C),
            ('B', 'C'): cliffs_delta(group_B, group_C)
        }

    # Effect sizes
    for pair, eff in effects.items():
        print(f"Effect size {pair}: {eff:.4f}")

    avg_magnitude, most_effective = find_most_effective_group(effects)
    print("\nAverage magnitudes:", avg_magnitude)
    print(Fore.CYAN + Style.BRIGHT + f"\nMost effective drink: {most_effective}" + Style.RESET_ALL)

    # Visualization: Pairwise Effect Sizes
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(map(str, effects.keys())), y=list(effects.values()), palette="coolwarm")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Pairwise Effect Sizes Between Drinks")
    plt.ylabel("Effect Size (Cohen's d / Cliff's delta)")
    plt.show()

else:
    print(Fore.CYAN + Style.BRIGHT + "\nNo significant difference between drinks." + Style.RESET_ALL)
