#INDFMPIR - poverty to income ratio
#DMDEDUC2 - education level
#DIQ010 - diabetes status
#RIDRETH3 - race
#RIAGENDR - biological sex
#RIDAGEYR - age

#data-binning for age
#histogram for age and sex
#chi-squared analysis on cross-tabs
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import numpy as np

demo_df = pd.read_sas('/content/DEMO_J.xpt', format='xport', encoding='utf-8')
diq_df = pd.read_sas('/content/DIQ_J.xpt', format='xport', encoding='utf-8')

merged_df = pd.merge(demo_df, diq_df, on='SEQN', how='inner')

#all variables
raw_vars = ['INDFMPIR', 'DMDEDUC2', 'DIQ010', 'RIDRETH3', 'RIAGENDR', 'RIDAGEYR']

#Filter for age
df_study = merged_df[(merged_df['RIDAGEYR'] >= 35) & (merged_df['RIDAGEYR'] <= 70)][raw_vars].copy()

#Clean data
df_study['DMDEDUC2'] = df_study['DMDEDUC2'].replace([7.0, 9.0], None)
df_study['DIQ010'] = df_study['DIQ010'].replace([7.0, 9.0], None).map({1: 1, 2: 0})

#descriptive statistics
print("INDFMPIR and RIDAGEYR description")
print(df_study[['INDFMPIR', 'RIDAGEYR']].describe())

print("\nDIQ010 (Diabetes Status) Counts")
print(df_study['DIQ010'].value_counts(dropna=False))

print("\nDIQ010 Percentages")
print(df_study['DIQ010'].value_counts(normalize=True) * 100)

print("\nDMDEDUC2 Percentages")
print(df_study['DMDEDUC2'].value_counts(normalize=True) * 100)

#cross-tab comparing DMDEDUC2 and DIQ010 (education)
print("\nDMDEDUC2 vs. DIQ010")
cross_edu_diab = pd.crosstab(
    df_study['DMDEDUC2'],
    df_study['DIQ010']
)
display(cross_edu_diab.round(2))

educhi = chi2_contingency(cross_edu_diab)
print("\n\nEducation vs diabetes chi squared analysis")
print(educhi)

# Logistic regression for eduation
model_edu = smf.logit(
    "DIQ010 ~ DMDEDUC2 + C(RIDRETH3) + C(RIAGENDR)",
    data=df_study
).fit()

results_edu = pd.DataFrame({
  "Odds Ratio": np.exp(model_edu.params),
  "Lower 95% CI": np.exp(model_edu.conf_int()[0]),
  "Upper 95% CI": np.exp(model_edu.conf_int()[1]),
  "p-value": model_edu.pvalues
})

print(results_edu)

#variable binning; separating age into buckets
df_study['age_bracket'] = pd.cut(
    df_study['RIDAGEYR'],
    bins=6,
    labels=['35-40', '41-46', '47-52', '53-58', '59-64', '65-70']
)
#variable binning; separating income into buckets
df_study['income_bracket'] = pd.cut(
    df_study['INDFMPIR'],
    bins=[0, 1.0, 2.0, 4.0, 10.0],
    labels=['Below Poverty (<1.0)', 'Low Income (1.0-1.99)', 'Middle Income (2.0-3.99)', 'High Income (4.0+)']
)

#Cross-tab comparing income brackets vs diabetes
print("\nCross-Tab: Income Bracket vs. Diabetes Diagnosis")
cross_income_diab = pd.crosstab(
    df_study['income_bracket'],
    df_study['DIQ010'],
)
display(cross_income_diab.round(2))

#chi squared analysis for income-poverty ratio
incomechi = chi2_contingency(cross_income_diab)
print("\n\nIncome vs diabetes chi squared analysis")
print(incomechi)

# Logistic regression for PIR
model_pir = smf.logit(
    "DIQ010 ~ INDFMPIR + C(RIDRETH3) + C(RIAGENDR)",
    data=df_study
).fit()

results_pir = pd.DataFrame({
  "Odds Ratio": np.exp(model_pir.params),
  "Lower 95% CI": np.exp(model_pir.conf_int()[0]),
  "Upper 95% CI": np.exp(model_pir.conf_int()[1]),
  "p-value": model_pir.pvalues
})

print(results_pir)



#Data-aggregation of the income buckets
print("\nPoverty Ratio by Diabetes Status")
income_by_diabetes = df_study.groupby('DIQ010')['INDFMPIR'].agg(
    Mean='mean',
    Median='median',
    Std_Dev='std',
    Sample_Size='count'
)
display(income_by_diabetes.round(2))
