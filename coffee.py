import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Set page config
st.set_page_config(page_title="Drink Effectiveness Study", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .big-font {
        font-size:18px !important;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)


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
    """Cliff's delta for non-parametric effect size."""
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
# 2. Streamlit App
# ===============================

def main():
    st.title("☕ Drink Effectiveness Study")
    st.markdown("""
    This app analyzes the effectiveness of different drinks (Regular Coffee, Decaf Coffee, and Energy Drink) 
    on productivity (measured by tasks completed per hour) using a within-subjects design.
    """)

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    n_participants = st.sidebar.slider("Number of Participants", 10, 100, 20)
    baseline_mean = st.sidebar.slider("Baseline Productivity (tasks/hour)", 30, 70, 50)
    baseline_std = st.sidebar.slider("Baseline Std Dev", 1, 10, 5)
    within_sd = st.sidebar.slider("Within-Subject Noise", 1.0, 5.0, 3.0)
    outlier_percent = st.sidebar.slider("Outlier Percentage", 0.0, 0.2, 0.07)
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05)

    # Simulate data
    np.random.seed(42)
    participants = np.arange(1, n_participants + 1)
    baseline = np.random.normal(loc=baseline_mean, scale=baseline_std, size=n_participants)

    # true (population) effects by drink (small differences)
    effects = {
        'Regular': 1.5,  # +1.5 tasks
        'Decaf': -0.5,  # -0.5 tasks
        'Energy': 3.0  # +3.0 tasks
    }

    # create long-format dataframe
    rows = []
    for i, subj in enumerate(participants):
        for drink, eff in effects.items():
            val = baseline[i] + eff + np.random.normal(0, within_sd)
            rows.append({'participant': subj, 'drink': drink, 'tasks': val})

    df = pd.DataFrame(rows)

    # inject outliers
    n_obs = len(df)
    n_outliers = max(1, int(n_obs * outlier_percent))
    outlier_idx = np.random.choice(df.index, size=n_outliers, replace=False)
    for idx in outlier_idx:
        df.at[idx, 'tasks'] += np.random.choice([20, -15])  # big jump

    # shuffle rows
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # separate each group
    group_A = df[df['drink'] == 'Regular']['tasks'].values
    group_B = df[df['drink'] == 'Decaf']['tasks'].values
    group_C = df[df['drink'] == 'Energy']['tasks'].values

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

    # ===============================
    # 3. Visualizations (EDA)
    # ===============================
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Boxplot with Swarmplot")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="drink", y="tasks", data=df, palette="Set2", ax=ax1)
        sns.swarmplot(x="drink", y="tasks", data=df, color=".25", ax=ax1)
        ax1.set_title("Distribution of Tasks Completed by Drink Type")
        st.pyplot(fig1)

    with col2:
        st.subheader("Violin Plot")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.violinplot(x="drink", y="tasks", data=df, palette="Set3", inner="quartile", ax=ax2)
        ax2.set_title("Violin Plot: Task Distribution per Drink")
        st.pyplot(fig2)

    # ===============================
    # 4. Assumption Checks
    # ===============================
    st.header("Statistical Assumption Checks")

    with st.expander("Normality Check (paired differences)"):
        diffs = {
            "Regular-Decaf": group_A - group_B,
            "Regular-Energy": group_A - group_C,
            "Decaf-Energy": group_B - group_C
        }

        normality_results = []
        for name, d in diffs.items():
            p_shapiro = stats.shapiro(d)[1]
            normality_results.append({
                "Comparison": name,
                "Shapiro-Wilk p-value": p_shapiro,
                "Normal?": p_shapiro > alpha
            })

        st.dataframe(pd.DataFrame(normality_results))

    with st.expander("Variance Homogeneity (Sphericity Proxy)"):
        levene_p = stats.levene(group_A, group_B, group_C)[1]
        box_color = "#e6f7ff"  # Very light blue background
        border_color = "#1890ff" if levene_p > alpha else "#ff4d4f"  # Blue/red border
        text_color = "#1890ff" if levene_p > alpha else "#ff4d4f"  # Blue/red text

        st.markdown(f"""
        <div style="background-color: {box_color}; 
                    border-left: 4px solid {border_color};
                    padding: 12px;
                    border-radius: 4px;
                    margin: 8px 0;">
            <p style="font-size: 16px; margin: 0; color: #333;">Levene's test p-value = <strong>{levene_p:.4f}</strong></p>
            <p style="font-size: 16px; margin: 8px 0 0 0; color: {text_color}; font-weight: 500;">
                Variances are {'equal' if levene_p > alpha else 'not equal'} (α = {alpha})
            </p>
        </div>
        """, unsafe_allow_html=True)

    all_normal = all(stats.shapiro(d)[1] > alpha for d in diffs.values())
    equal_var = levene_p > alpha

    # ===============================
    # 5. Hypothesis Testing
    # ===============================
    st.header("Hypothesis Testing Results")

    if all_normal and equal_var:
        test_name = "Repeated-measures ANOVA"
        stat, p_value = stats.f_oneway(group_A, group_B, group_C)
    else:
        test_name = "Friedman Test"
        stat, p_value = stats.friedmanchisquare(group_A, group_B, group_C)

    result_box_color = "#f6ffed"  # Very light green background
    result_border = "#52c41a" if p_value < alpha else "#ff4d4f"  # Green/red border
    result_text = "#52c41a" if p_value < alpha else "#ff4d4f"  # Green/red text

    st.markdown(f"""
    <div style="background-color: {result_box_color}; 
                border-left: 4px solid {result_border};
                padding: 12px;
                border-radius: 4px;
                margin: 12px 0;">
        <p style="font-size: 16px; margin: 0; color: #333;">Test used: <strong>{test_name}</strong></p>
        <p style="font-size: 16px; margin: 8px 0; color: #333;">
            Test statistic = <strong>{stat:.4f}</strong>, p-value = <strong>{p_value:.4f}</strong>
        </p>
        <p style="font-size: 16px; margin: 0; color: {result_text}; font-weight: 500;">
            Result: {'Significant' if p_value < alpha else 'Not significant'} (α = {alpha})
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # 6. Post-hoc Analysis & Effect Size
    # ===============================
    if p_value < alpha:
        st.success("Significant differences detected between drink types!")

        if test_name == "Repeated-measures ANOVA":
            effects = {
                ('Regular', 'Decaf'): cohens_d_paired(group_A, group_B),
                ('Regular', 'Energy'): cohens_d_paired(group_A, group_C),
                ('Decaf', 'Energy'): cohens_d_paired(group_B, group_C)
            }
            effect_name = "Cohen's d"
        else:
            effects = {
                ('Regular', 'Decaf'): cliffs_delta(group_A, group_B),
                ('Regular', 'Energy'): cliffs_delta(group_A, group_C),
                ('Decaf', 'Energy'): cliffs_delta(group_B, group_C)
            }
            effect_name = "Cliff's delta"

        # Display effect sizes
        st.subheader(f"Pairwise Effect Sizes ({effect_name})")

        effect_df = pd.DataFrame([{"Group 1": k[0], "Group 2": k[1], "Effect Size": v}
                                  for k, v in effects.items()])
        st.dataframe(effect_df.style.format({"Effect Size": "{:.4f}"}))

        # Find most effective group
        avg_magnitude, most_effective = find_most_effective_group(effects)
        st.success(f"Most effective drink: {most_effective}")

        # Visualization: Pairwise Effect Sizes
        st.subheader("Effect Size Visualization")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(map(str, effects.keys())), y=list(effects.values()), palette="coolwarm", ax=ax3)
        ax3.axhline(0, color="black", linestyle="--")
        ax3.set_title(f"Pairwise Effect Sizes Between Drinks ({effect_name})")
        ax3.set_ylabel("Effect Size")
        st.pyplot(fig3)
    else:
        st.info("No significant difference detected between drink types.")


if __name__ == "__main__":
    main()