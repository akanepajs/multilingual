import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from IPython.display import display


# Let's first load the CSV file the user uploaded and summarize its contents

# Reading the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/akanepajs/multilingual/main/Data.csv'

# How many observations per country per prompt type?
n_total = 100

# Load the data into a pandas DataFrame
data_df = pd.read_csv(file_path)

# Append the log(CommonCrawl corpus, %) column
data_df['Log CommonCrawl corpus (%)'] = np.log(data_df['CommonCrawl corpus (%)'])

# Define the list of models in alphabetical order
models_list = sorted(data_df['Model'].unique())

# Initialize a storage structure for the regression results
results_dict = {
    'Model': [],
    'Dependent Variable': [],
    'Intercept': [],
    'Slope': [],
    'p-value': []
}

# Loop through each model and perform OLS regressions for both Harmful Accepted and Harmless Rejected
for model_name in models_list:
    # Filter the data for the specific model
    model_df = data_df[data_df['Model'] == model_name]

    # Harmful Accepted OLS
    X_harmful = sm.add_constant(model_df['Log CommonCrawl corpus (%)'])
    y_harmful = model_df['Harmful ACCEPTED']
    ols_model_harmful = sm.OLS(y_harmful, X_harmful).fit()

    # Store the results for Harmful Accepted
    results_dict['Model'].append(model_name)
    results_dict['Dependent Variable'].append('Harmful ACCEPTED')
    results_dict['Intercept'].append(ols_model_harmful.params['const'])
    results_dict['Slope'].append(ols_model_harmful.params['Log CommonCrawl corpus (%)'])
    results_dict['p-value'].append(ols_model_harmful.pvalues['Log CommonCrawl corpus (%)'])

    # Harmless Rejected OLS
    X_harmless = sm.add_constant(model_df['Log CommonCrawl corpus (%)'])
    y_harmless = model_df['Harmless REJECTED']
    ols_model_harmless = sm.OLS(y_harmless, X_harmless).fit()

    # Store the results for Harmless Rejected
    results_dict['Model'].append(model_name)
    results_dict['Dependent Variable'].append('Harmless REJECTED')
    results_dict['Intercept'].append(ols_model_harmless.params['const'])
    results_dict['Slope'].append(ols_model_harmless.params['Log CommonCrawl corpus (%)'])
    results_dict['p-value'].append(ols_model_harmless.pvalues['Log CommonCrawl corpus (%)'])

# Convert the results dictionary into a DataFrame for easy viewing
regression_results_df = pd.DataFrame(results_dict)


# Initialize a storage structure for the logistic regression results
logit_results_dict = {
    'Model': [],
    'Dependent Variable': [],
    'Intercept': [],
    'Slope': [],
    'p-value': []
}

# Loop through each model and perform logistic regressions for both Harmful Accepted and Harmless Rejected
for model_name in models_list:
    # Filter the data for the specific model
    model_df = data_df[data_df['Model'] == model_name]

    # Initialize lists to store the expanded data for Harmful Accepted and Harmless Rejected
    expanded_log_corpus_harmful = []
    expanded_outcome_harmful = []
    expanded_log_corpus_harmless = []
    expanded_outcome_harmless = []

    # Expand the data to 100 binary observations per language for logistic regression
    for index, row in model_df.iterrows():
        expanded_log_corpus_harmful.extend([row['Log CommonCrawl corpus (%)']] * row['Harmful ACCEPTED'])
        expanded_outcome_harmful.extend([1] * row['Harmful ACCEPTED'])
        expanded_log_corpus_harmful.extend([row['Log CommonCrawl corpus (%)']] * row['Harmful REJECTED'])
        expanded_outcome_harmful.extend([0] * row['Harmful REJECTED'])
        expanded_log_corpus_harmful.extend([row['Log CommonCrawl corpus (%)']] * row['Harmful UNCLEAR'])
        expanded_outcome_harmful.extend([0] * row['Harmful UNCLEAR'])

        expanded_log_corpus_harmless.extend([row['Log CommonCrawl corpus (%)']] * row['Harmless REJECTED'])
        expanded_outcome_harmless.extend([1] * row['Harmless REJECTED'])
        expanded_log_corpus_harmless.extend([row['Log CommonCrawl corpus (%)']] * row['Harmless ACCEPTED'])
        expanded_outcome_harmless.extend([0] * row['Harmless ACCEPTED'])
        expanded_log_corpus_harmless.extend([row['Log CommonCrawl corpus (%)']] * row['Harmless UNCLEAR'])
        expanded_outcome_harmless.extend([0] * row['Harmless UNCLEAR'])

    # Convert expanded data to DataFrame
    expanded_df_harmful = pd.DataFrame({
        'Log CommonCrawl corpus (%)': expanded_log_corpus_harmful,
        'Outcome': expanded_outcome_harmful
    })

    expanded_df_harmless = pd.DataFrame({
        'Log CommonCrawl corpus (%)': expanded_log_corpus_harmless,
        'Outcome': expanded_outcome_harmless
    })



    # Check data:
    #display(expanded_df_harmful)
    #expanded_df_harmless.to_excel('expanded_df_harmless.xlsx', index=False)


    # Harmful Accepted Logit Regression
    X_logit_harmful = sm.add_constant(expanded_df_harmful['Log CommonCrawl corpus (%)'])
    y_logit_harmful = expanded_df_harmful['Outcome']
    logit_model_harmful = sm.Logit(y_logit_harmful, X_logit_harmful).fit(disp=0)

    # Store the results for Harmful Accepted
    logit_results_dict['Model'].append(model_name)
    logit_results_dict['Dependent Variable'].append('Harmful ACCEPTED')
    logit_results_dict['Intercept'].append(logit_model_harmful.params['const'])
    logit_results_dict['Slope'].append(logit_model_harmful.params['Log CommonCrawl corpus (%)'])
    logit_results_dict['p-value'].append(logit_model_harmful.pvalues['Log CommonCrawl corpus (%)'])

    # Harmless Rejected Logit Regression
    X_logit_harmless = sm.add_constant(expanded_df_harmless['Log CommonCrawl corpus (%)'])
    y_logit_harmless = expanded_df_harmless['Outcome']
    logit_model_harmless = sm.Logit(y_logit_harmless, X_logit_harmless).fit(disp=0)

    # Store the results for Harmless Rejected
    logit_results_dict['Model'].append(model_name)
    logit_results_dict['Dependent Variable'].append('Harmless REJECTED')
    logit_results_dict['Intercept'].append(logit_model_harmless.params['const'])
    logit_results_dict['Slope'].append(logit_model_harmless.params['Log CommonCrawl corpus (%)'])
    logit_results_dict['p-value'].append(logit_model_harmless.pvalues['Log CommonCrawl corpus (%)'])

# Convert the results dictionary into a DataFrame for easy viewing
logit_results_df = pd.DataFrame(logit_results_dict)


# FORMATTING TABLES TO LATEX

def format_coefficient(coef):
    """Format coefficient to 2 decimal places"""
    return f"{coef:.2f}"

def format_pvalue(p_value):
    """Format p-value to 3 decimal places"""
    return f"{p_value:.3f}"

def get_significance_stars(p_value):
    """Return significance stars based on p-value"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""

def create_latex_tables(logit_results_df, regression_results_df):
    # Process logistic regression results
    models = sorted(logit_results_df['Model'].unique())
    
    # Adjust p-values using Benjamini-Hochberg method
    p_values = logit_results_df['p-value'].values
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
    logit_results_df['p_adj'] = adjusted_p_values
    
    # Create LaTeX table for logistic regression
    logit_table = "\\begin{table}[H]\n\\centering\n"
    logit_table += "\\caption{Logistic regression results: relationship between dependent variable and log(CommonCrawl corpus share). "\
                   "2400 observations across 24 languages (100 per language) for each regression, with Benjamini--Hochberg adjusted p-values. "\
                   "Adjusted significance levels: *, **, *** represent 5\\%, 1\\%, and 0.1\\%, respectively.}\n"
    logit_table += "\\label{Table2}\n"
    logit_table += "\\begin{tabular}{lcccc cccc}\n\\toprule\n"
    logit_table += "\\multirow{4}{*}{Model} & \\multicolumn{4}{c}{Harmful Accepted} & \\multicolumn{4}{c}{Harmless Rejected} \\\\\n"
    logit_table += "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
    logit_table += " & $\\beta_0$ & $\\beta_1$ & $p_\\textrm{adj}$ & Sig. & $\\beta_0$ & $\\beta_1$ & $p_\\textrm{adj}$ & Sig. \\\\\n"
    logit_table += "\\midrule\n"

    for model in models:
        harmful = logit_results_df[(logit_results_df['Model'] == model) & 
                                 (logit_results_df['Dependent Variable'] == 'Harmful ACCEPTED')].iloc[0]
        harmless = logit_results_df[(logit_results_df['Model'] == model) & 
                                  (logit_results_df['Dependent Variable'] == 'Harmless REJECTED')].iloc[0]
        
        row = f"{model} & {format_coefficient(harmful['Intercept'])} & {format_coefficient(harmful['Slope'])} & "
        row += f"{format_pvalue(harmful['p_adj'])} & {get_significance_stars(harmful['p_adj'])} & "
        row += f"{format_coefficient(harmless['Intercept'])} & {format_coefficient(harmless['Slope'])} & "
        row += f"{format_pvalue(harmless['p_adj'])} & {get_significance_stars(harmless['p_adj'])} \\\\\n"
        logit_table += row

    logit_table += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Process linear regression results similarly
    p_values = regression_results_df['p-value'].values
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
    regression_results_df['p_adj'] = adjusted_p_values

    # Create LaTeX table for linear regression
    linear_table = "\\begin{table}[H]\n\\centering\n"
    linear_table += "\\caption{Linear regression results: relationship between dependent variable and log(CommonCrawl corpus share). "\
                    "Benjamini--Hochberg adjusted p-values. "\
                    "Adjusted significance levels: *, **, *** represent 5\\%, 1\\%, and 0.1\\%, respectively.}\n"
    linear_table += "\\label{Table3}\n"
    linear_table += "\\begin{tabular}{lcccc cccc}\n\\toprule\n"
    linear_table += "\\multirow{4}{*}{Model} & \\multicolumn{4}{c}{Harmful Accepted} & \\multicolumn{4}{c}{Harmless Rejected} \\\\\n"
    linear_table += "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
    linear_table += " & $\\beta_0$ & $\\beta_1$ & $p_\\textrm{adj}$ & Sig. & $\\beta_0$ & $\\beta_1$ & $p_\\textrm{adj}$ & Sig. \\\\\n"
    linear_table += "\\midrule\n"

    for model in models:
        harmful = regression_results_df[(regression_results_df['Model'] == model) & 
                                     (regression_results_df['Dependent Variable'] == 'Harmful ACCEPTED')].iloc[0]
        harmless = regression_results_df[(regression_results_df['Model'] == model) & 
                                      (regression_results_df['Dependent Variable'] == 'Harmless REJECTED')].iloc[0]
        
        row = f"{model} & {format_coefficient(harmful['Intercept'])} & {format_coefficient(harmful['Slope'])} & "
        row += f"{format_pvalue(harmful['p_adj'])} & {get_significance_stars(harmful['p_adj'])} & "
        row += f"{format_coefficient(harmless['Intercept'])} & {format_coefficient(harmless['Slope'])} & "
        row += f"{format_pvalue(harmless['p_adj'])} & {get_significance_stars(harmless['p_adj'])} \\\\\n"
        linear_table += row

    linear_table += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return logit_table, linear_table

# Generate and print the tables
logit_table, linear_table = create_latex_tables(logit_results_df, regression_results_df)
print("Logistic Regression Table:")
print(logit_table)
print("\nLinear Regression Table:")
print(linear_table)






### GENERATING PLOTS
model_name = 'Mistral Large 2' # 'GPT-4o', 'Claude 3.5 Sonnet', 'Gemini 1.5 Pro', 'Llama 3.1 405B', 'Mistral Large 2'
dep_var= 'Harmful ACCEPTED'   # 'Harmful ACCEPTED','Harmless REJECTED'

# Filter data for the selected model
model_df = data_df[data_df['Model'] == model_name]

# Extract OLS results for the model (from regression_results_df)
ols_results = regression_results_df[regression_results_df['Model'] == model_name]
ols_intercept = ols_results[ols_results['Dependent Variable'] == dep_var]['Intercept'].values[0]
ols_slope = ols_results[ols_results['Dependent Variable'] == dep_var]['Slope'].values[0]
ols_pvalue = ols_results[ols_results['Dependent Variable'] == dep_var]['p-value'].values[0]

# Extract Logit results for the model (from logit_results_df)
logit_results = logit_results_df[logit_results_df['Model'] == model_name]
logit_intercept = logit_results[logit_results_df['Dependent Variable'] == dep_var]['Intercept'].values[0]
logit_slope = logit_results[logit_results_df['Dependent Variable'] == dep_var]['Slope'].values[0]
logit_pvalue = logit_results[logit_results_df['Dependent Variable'] == dep_var]['p-value'].values[0]

# Calculate the OLS line and logistic curve for the plot
sorted_commoncrawl_corpus = np.sort(model_df['CommonCrawl corpus (%)'])
ols_line = ols_intercept + ols_slope * np.log(sorted_commoncrawl_corpus)
logit_curve = 1 / (1 + np.exp(-(logit_intercept + logit_slope * np.log(sorted_commoncrawl_corpus))))

# Using the observed data from the model for error bars
observed_proportions = model_df[dep_var] / n_total

# Function to calculate the Wilson score interval
def wilson_score_interval(successes, n, alpha=0.05):
    lower_bounds, upper_bounds = proportion_confint(successes, n, alpha=alpha, method='wilson')
    return lower_bounds, upper_bounds

# Calculate the Wilson score interval for each observation
lower_bounds, upper_bounds = wilson_score_interval(model_df[dep_var], n_total)

# Plot the figure with observed data, logistic curve, and OLS line, using Wilson score intervals
plt.figure(figsize=(10, 6))
plt.errorbar(model_df['CommonCrawl corpus (%)'], observed_proportions * 100,
             yerr=[(observed_proportions - lower_bounds) * 100, (upper_bounds - observed_proportions) * 100],
             fmt='o', color='blue', ecolor='lightgray', elinewidth=2, capsize=3, label='Observed proportion with 95% CI (Wilson score interval)')

# Plot the logistic regression curve
plt.plot(sorted_commoncrawl_corpus, logit_curve * 100, color='red',
         label=r'Logistic regression, 2400 obs.: $P = \frac{1}{1 + e^{-(%.2f + %.2f \log(x))}}$' % (logit_intercept, logit_slope) +
               '\n' + 'slope coefficient p-value = %.3f' % logit_pvalue)

# Plot the OLS regression line
plt.plot(sorted_commoncrawl_corpus, ols_line, color='green', linestyle='--',
         label=r'OLS regression, 24 obs.: $y = %.2f + %.2f \log(x)$' % (ols_intercept, ols_slope) +
               '\n' + 'slope coefficient p-value = %.3f' % ols_pvalue)

# Add language code labels to the data points
for i, txt in enumerate(model_df['Language_short']):
    plt.annotate(txt, (model_df['CommonCrawl corpus (%)'].iloc[i], observed_proportions.iloc[i] * 100),
                 textcoords="offset points", xytext=(-7, 0), ha='right', fontsize=12)

# Customize axes and labels
#plt.xlim(0.000001, 10000)
#plt.ylim(10, 100)
plt.xscale('log')
plt.xlabel('CommonCrawl Corpus (%), log scale')
plt.ylabel('%')
plt.xticks([0.01, 0.1, 1, 10], ['0.01%', '0.1%', '1%', '10%'])
#plt.title(f'{model_name}, Harmful Accepted Proportion (OLS vs. Logistic Regression)')
plt.legend(loc='best')
plt.grid(True, color='gray', alpha=0.2)

# Show the plot with updated Wilson score intervals
plt.show()


