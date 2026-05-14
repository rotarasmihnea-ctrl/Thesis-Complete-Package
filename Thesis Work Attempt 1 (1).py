#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import csv
import re
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade openai')

from openai import OpenAI

# ---------------------------------------------------------
# ---------------------------------------------------------

# !! - Upon an initial run in the JUPYTERLAB environment, a number of packages will be imported, followed by an error message:

#Error Name : ImportError: cannot import name 'Sentinel' from 'typing_extensions' (/opt/conda/lib/python3.10/site-packages/typing_extensions.py)

# This is normal, once the error message appears and the program stops, the kernel should be manually restarted and then the program should run as intended.

# ---------------------------------------------------------
# ---------------------------------------------------------


data = "HUMOD_Dataset.TXT"
nrsample = 200
model = "gpt-4.1-mini"
output = "sample_for_annotation_200_mental_states-RESULTS.csv"


# ---------------------------------------------------------
# STEP 1 — LOADING THE DATASET
# ---------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath,sep="\t",encoding="utf-8",engine="python",quoting=csv.QUOTE_MINIMAL)
    df.columns = [col.strip() for col in df.columns]
    return df

full_df = load_dataset(data)
print("Dataset shape:", full_df.shape)

# ---------------------------------------------------------
# STEP 2 — KEEPING ONLY THE RELEVANT DATASET COLUMNS
# ---------------------------------------------------------

def relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Dialogue_ID","Dialogue_Context","Reply","Label","User_Relevance_Score","User_Reply"]
    return df[columns].copy()

selected_df = relevant_columns(full_df)
print("Dataset shape (with relevant columns only):", selected_df.shape)

# --------------------------------------------------------------------
# STEP 3 — AGREGGATE HUMAN SCORES BY GENERATING THE MEAN HUMAN SCORES
# --------------------------------------------------------------------

def mean_human_scores(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (df.groupby(["Dialogue_ID", "Dialogue_Context", "Reply", "Label"], as_index=False).agg(mean_human_score=("User_Relevance_Score", "mean"),
    n_ratings=("User_Relevance_Score", "count")))
    return grouped

aggregated_df = mean_human_scores(selected_df)
print("Dataset shape (with mean human scores):", aggregated_df.shape)

# ---------------------------------------------------------
# STEP 4 — CREATE A 200-CONTEXT SAMPLE DATASET
# ---------------------------------------------------------

def sample_dataset(df: pd.DataFrame, n_sample: int = 200, random_state: int = 42) -> pd.DataFrame:
    
    sample_df = df.sample(n=n_sample, random_state=random_state).reset_index(drop=True)
    return sample_df
    
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
# THE "STEP 4" CODE WAS USED TO CREATE THE 200-SAMPLE DATASET, WHICH RECEIVED MENTAL STATE DATA MANUALLY AFTER CREATION
# THE RESULTING FILE IS "sample_for_annotation_200_mental_states.csv" AND IT IS PRESENT IN THE CODE PACKAGE
# THE FILE IS FIXED AND WILL ALTER THE RESULTS IF REGENERATED

# -- ! THE 2 LINES OF CODE BELLOW SHOULD NOT BE UNCOMMENTED ! -- (THEY ARE HERE TO SHOWCASE HOW THE FILE WAS INITIALLY CREATED)

#sample_df_200 = sample_dataset(aggregated_df, n_sample=200)
#sample_df_200.to_csv("sample_for_annotation_200.csv", index=False)
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------
# STEP 5 — SET UP THE MODEL CLIENT (API KEY INCLUDED) 
# ---------------------------------------------------------



def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# STEP 6 — LOAD THE ANNOTATED SAMPLE DATASET
# ---------------------------------------------------------

def load_annotated_sample(filepath: str) -> pd.DataFrame:
    
    df = pd.read_csv(filepath, encoding="utf-8")
    df.columns = [col.strip() for col in df.columns]

    df["mental_state"] = df["mental_state"].astype(str).str.strip(' ",')

    assert "mental_state" in df.columns, "mental_state column is missing!"
    assert df["mental_state"].notna().all(), "Some mental_state values are missing!"
    return df


sample_df = load_annotated_sample("sample_for_annotation_200_mental_states.csv")
print("Annotated sample shape:", sample_df.shape)

sample_df[["Dialogue_ID", "Dialogue_Context", "Reply", "Label", "mean_human_score", "n_ratings", "mental_state"]].head(200)

# ---------------------------------------------------------
# STEP 7 — CREATE A HELPER FUNCTION
# (IN ORDER TO ONLY EXTRACT THE SCORE GIVEN BY THE MODEL)
# ---------------------------------------------------------

def extract_score(text: str):
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    return None

# ---------------------------------------------------------
# STEP 8 — INTRODUCE THE BASELINE PROMPT TO THE MODEL
# ---------------------------------------------------------

def get_model_score_baseline(client: OpenAI, context: str, reply: str, model_name: str) -> int | None:
    prompt = f"""
You are evaluating dialogue relevance.

Dialogue context:
{context}

Candidate reply:
{reply}

Task:
Rate how contextually appropriate the candidate reply is on a scale from 1 to 5.

Scale:
1 = very poor fit
2 = poor fit
3 = moderate fit
4 = good fit
5 = excellent fit

Return only one number: 1, 2, 3, 4, or 5.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful evaluator of dialogue relevance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    print("Raw baseline response:", text)
    return extract_score(text)

# ----------------------------------------------------------
# STEP 9 — INTRODUCE THE THEORY OF MIND PROMPT TO THE MODEL
# ----------------------------------------------------------

def get_model_score_tom(client: OpenAI, context: str, reply: str, mental_state: str, model_name: str) -> int | None:
    prompt = f"""
You are evaluating dialogue relevance.

Dialogue context:
{context}

Inferred speaker mental or emotional state:
{mental_state}

Candidate reply:
{reply}

Task:
Rate how contextually appropriate the candidate reply is on a scale from 1 to 5.

Scale:
1 = very poor fit
2 = poor fit
3 = moderate fit
4 = good fit
5 = excellent fit

Return only one number: 1, 2, 3, 4, or 5.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful evaluator of dialogue relevance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    print("Raw ToM response:", text)
    return extract_score(text)

# ---------------------------------------------------------
# STEP 10 — RUN THE MODEL OVER THE SAMPLED DATASET
# (BOTH CONDITIONS ARE RUN FOR EACH ROW IN THE SAMPLE)
# ---------------------------------------------------------

def run_model_scoring(sample_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    client = get_client()
    df = sample_df.copy()
    baseline_scores = []
    tom_scores = []
    for idx, row in df.iterrows():
        print(f"Scoring item {idx + 1}/{len(df)}")
        baseline_score = get_model_score_baseline(
            client=client,
            context=row["Dialogue_Context"],
            reply=row["Reply"],
            model_name=model_name
        )
        print(f"   Baseline score: {baseline_score}")
        tom_score = get_model_score_tom(
            client=client,
            context=row["Dialogue_Context"],
            reply=row["Reply"],
            mental_state=row["mental_state"],
            model_name=model_name
        )
        print(f"   ToM score: {tom_score}")
        baseline_scores.append(baseline_score)
        tom_scores.append(tom_score)
    df["model_score_baseline"] = baseline_scores
    df["model_score_tom"] = tom_scores
    return df


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# --- ! RUN THIS ONLY ONCE ! ---
# (THIS IS RUN ONLY ONCE IN ORDER TO ACHIEVE MODEL SCORES FOR EACH ROW AND SAVE THEM TO THE OUTPUT FILE)
# (RUNNING THE NEXT TWO LINES EVERY TIME, IN ADDITION TO THE ONE AFTER, WILL RESULT IN SLIGHT VARIATIONS IN RESULTS FOR EACH RUN)

#scored_df = run_model_scoring(sample_df, model)
#scored_df.to_csv(output, index=False)

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# --- ! USE THIS FOR ALL FUTURE RUNS ! ---

scored_df = pd.read_csv(output)
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


scored_df[["Dialogue_ID", "mean_human_score", "model_score_baseline", "model_score_tom"]].head(200)

# ---------------------------------------------------------
# STEP 11 — SAVE RESULTS
# ---------------------------------------------------------

def save_results(df: pd.DataFrame, output_path: str):
    """
    Saves results so you do not lose progress.
    """
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nResults saved to: {output_path}")



save_results(scored_df, output)

# -----------------------------------------------------------
# STEP 12 — ANALYSIS: PEARSON'S CORRELATION (INCLUDING PLOT)
# -----------------------------------------------------------

def run_Pearson_correlation(df: pd.DataFrame):
    clean_baseline = df.dropna(subset=["mean_human_score", "model_score_baseline"])
    clean_tom = df.dropna(subset=["mean_human_score", "model_score_tom"])
    r_baseline, p_baseline = pearsonr(
        clean_baseline["mean_human_score"],
        clean_baseline["model_score_baseline"]
    )
    r_tom, p_tom = pearsonr(
        clean_tom["mean_human_score"],
        clean_tom["model_score_tom"]
    )
    print(f"Baseline: r={r_baseline:.3f}, p={p_baseline:.5f}")
    print(f"Theory-of-Mind: r={r_tom:.3f}, p={p_tom:.5f}")
    plt.figure(figsize=(6, 5))

    conditions = ["Baseline", "Theory of Mind"]
    correlations = [r_baseline, r_tom]
    colors = ["#7A0019", "green"]

    plt.bar(conditions, correlations, color=colors)

    plt.ylabel("Pearson Correlation with Human Scores")
    plt.ylim(0, 1)
    plt.title("Human–Model Correlation by Condition")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

run_Pearson_correlation(scored_df)



# ---------------------------------------------------------
# STEP 13 — ANALYSIS: MEAN SQUARED ERROR (INCLUDING PLOT)
# ---------------------------------------------------------

from sklearn.metrics import mean_squared_error

def compute_mse(df: pd.DataFrame):
    baseline_mse = mean_squared_error(df["mean_human_score"], df["model_score_baseline"])
    tom_mse = mean_squared_error(df["mean_human_score"], df["model_score_tom"])

    print("\n=== MEAN SQUARED ERROR ===")
    print(f"Baseline MSE: {baseline_mse:.3f}")
    print(f"Theory-of-Mind MSE: {tom_mse:.3f}")

    return baseline_mse, tom_mse

baseline_mse, tom_mse = compute_mse(scored_df)

plt.figure(figsize=(6, 5))

conditions = ["Baseline", "Theory of Mind"]
mse_values = [baseline_mse, tom_mse]
colors = ["#7A0019", "green"]

plt.bar(conditions, mse_values, color=colors)

plt.ylabel("Mean Squared Error")
plt.title("Prediction Error by Condition")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# STEP 14 — ANALYSIS: PAIRED T-TEST 
# ---------------------------------------------------------

from scipy.stats import ttest_rel

def compute_ttest(df: pd.DataFrame):
    # Compute absolute errors
    baseline_errors = abs(df["mean_human_score"] - df["model_score_baseline"])
    tom_errors = abs(df["mean_human_score"] - df["model_score_tom"])

    t_stat, p_value = ttest_rel(baseline_errors, tom_errors)

    print("\n=== PAIRED T-TEST (ERROR COMPARISON) ===")
    print(f"t = {t_stat:.3f}")
    print(f"p = {p_value:.5f}")

    return t_stat, p_value

t_stat, p_value = compute_ttest(scored_df)


# ---------------------------------------------------------
# STEP 15 — ANALYSIS: LINEAR REGRESSION
# (USING: "human_score ~ model_score * condition")
# ---------------------------------------------------------

def run_regression_analysis(df: pd.DataFrame):
    """
    Runs linear regression using:
    human_score ~ model_score * condition

    WHY THIS MATTERS:
    Correlation tells you whether scores move together.
    Regression lets you test whether the ToM condition
    changes the relationship between model and human scores.
    """
    baseline_df = df[["Dialogue_ID", "mean_human_score", "model_score_baseline"]].copy()
    baseline_df["condition"] = "baseline"
    baseline_df = baseline_df.rename(columns={"model_score_baseline": "model_score"})

    tom_df = df[["Dialogue_ID", "mean_human_score", "model_score_tom"]].copy()
    tom_df["condition"] = "tom"
    tom_df = tom_df.rename(columns={"model_score_tom": "model_score"})

    analysis_df = pd.concat([baseline_df, tom_df], ignore_index=True)
    analysis_df = analysis_df.dropna()

    model = smf.ols(
        "mean_human_score ~ model_score * condition",
        data=analysis_df
    ).fit()

    print("\n=== REGRESSION RESULTS ===")
    print(model.summary())

run_regression_analysis(scored_df)

# --------------------------------------------------------------------------------------
# STEP 16 — SECONDARY ANALYSIS: BINARY CLASSIFICATION

# (THIS TESTS WHETHER THE MODEL CAN DISTINGUISH APPROPRIATE FROM INAPPROPRIATE REPLIES)
# (THRESHOLD: score >= 3 -> positive; score < 3 -> negative)
# --------------------------------------------------------------------------------------

def run_classification_analysis(df: pd.DataFrame):
    
    clean_df = df.dropna(subset=["model_score_baseline", "model_score_tom"]).copy()

    clean_df["pred_label_baseline"] = (clean_df["model_score_baseline"] >= 3).astype(int)
    clean_df["pred_label_tom"] = (clean_df["model_score_tom"] >= 3).astype(int)

    acc_b = accuracy_score(clean_df["Label"], clean_df["pred_label_baseline"])
    prec_b = precision_score(clean_df["Label"], clean_df["pred_label_baseline"], zero_division=0)
    rec_b = recall_score(clean_df["Label"], clean_df["pred_label_baseline"], zero_division=0)

    acc_t = accuracy_score(clean_df["Label"], clean_df["pred_label_tom"])
    prec_t = precision_score(clean_df["Label"], clean_df["pred_label_tom"], zero_division=0)
    rec_t = recall_score(clean_df["Label"], clean_df["pred_label_tom"], zero_division=0)

    print("\n=== CLASSIFICATION RESULTS ===")
    print(f"Baseline -> Accuracy: {acc_b:.3f}, Precision: {prec_b:.3f}, Recall: {rec_b:.3f}")
    print(f"ToM      -> Accuracy: {acc_t:.3f}, Precision: {prec_t:.3f}, Recall: {rec_t:.3f}")


run_classification_analysis(scored_df)

# --------------------------------------------------------------------------------------
# STEP 17 — PLOT A SCATTER PLOT
# (IN ORDER TO SHOWCASE THE HUMAN-MODEL ALIGNMENT IN DIALOGUE RELEVANCE)
# --------------------------------------------------------------------------------------


jitter_strength = 0.08
x_human = scored_df["mean_human_score"]
x_baseline = x_human + np.random.normal(0, jitter_strength, len(scored_df))
y_baseline = scored_df["model_score_baseline"] + np.random.normal(0, jitter_strength, len(scored_df))
x_tom = x_human + np.random.normal(0, jitter_strength, len(scored_df))
y_tom = scored_df["model_score_tom"] + np.random.normal(0, jitter_strength, len(scored_df))
plt.figure(figsize=(7, 6))
plt.scatter(x_baseline, y_baseline, color="#7A0019", alpha=0.6, label="Baseline")
plt.scatter(x_tom, y_tom, color="green", alpha=0.6, label="Theory of Mind")

coef_baseline = np.polyfit(scored_df["mean_human_score"], scored_df["model_score_baseline"], 1)
coef_tom = np.polyfit(scored_df["mean_human_score"], scored_df["model_score_tom"], 1)
x_line = np.linspace(1, 5, 100)
y_line_baseline = coef_baseline[0] * x_line + coef_baseline[1]
y_line_tom = coef_tom[0] * x_line + coef_tom[1]

plt.plot(x_line, y_line_baseline, color="#7A0019", linewidth=2)
plt.plot(x_line, y_line_tom, color="green", linewidth=2)
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3, 4, 5])
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel("Mean Human Relevance Score")
plt.ylabel("Model Relevance Score")
plt.title("Human vs Model Scores with Regression Lines")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:




