

## 📌 Project Overview
This project investigates **whether different types of drinks (Regular Coffee, Decaf, and Energy Drinks) affect productivity**, measured by the number of tasks completed.  

The workflow follows a **data analyst’s approach**:
1. **Exploratory Data Analysis (EDA)** with visualizations.  
2. **Assumption checks** (normality & variance).  
3. **Statistical hypothesis testing** (ANOVA or Friedman).  
4. **Post-hoc effect size analysis** (Cohen’s *d* or Cliff’s Delta).  
5. **Clear visualization of results** to support conclusions.  

---

## 📊 Dataset
The dataset is **synthetic** but designed to mimic a productivity experiment with 10 participants tested under three conditions:
- **Regular Coffee**  
- **Decaf Coffee**  
- **Energy Drink**

Example (long format):

| subject | drink    | tasks |
|---------|----------|-------|
| 1       | Regular  | 12    |
| 2       | Regular  | 14    |
| ...     | ...      | ...   |
| 1       | Decaf    | 11    |
| ...     | ...      | ...   |
| 1       | Energy   | 16    |

---

## 🔬 Hypotheses
**Null hypothesis (H₀):** There is no difference in productivity across drinks.  
**Alternative hypothesis (H₁):** At least one drink leads to significantly different productivity.  

---

## ⚙️ Methodology
1. **Normality check**  
   - Shapiro-Wilk (paired differences)  
   - Kolmogorov-Smirnov (with sample mean/std)  

2. **Equal variance check**  
   - Levene’s Test  

3. **Choice of test**  
   - If assumptions hold → **Repeated-measures ANOVA**  
   - If not → **Friedman Test**  

4. **Post-hoc effect size**  
   - **Cohen’s d** for ANOVA  
   - **Cliff’s Delta** for Friedman  

---

## 📈 Visualizations
- **Boxplot + swarmplot** → distribution & individual data points.  
- **Violin plot** → density and spread.  
- **Effect size barplot** → strength of differences between groups.  

---

## ✅ Results (example run)
- **Test used:** Repeated-measures ANOVA  
- **p-value:** < 0.001 → Reject H₀  
- **Conclusion:** Energy drinks led to significantly higher productivity.  
- **Effect sizes:**  
  - A vs B (Regular vs Decaf): medium effect  
  - A vs C (Regular vs Energy): large effect  
  - B vs C (Decaf vs Energy): large effect  

📊 **Most effective drink: Energy**  

---

## 📦 Installation & Usage
```bash
# Clone repo
git clone https://github.com/yourusername/drink-productivity-analysis.git
cd drink-productivity-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python analysis.py
