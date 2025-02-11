# Creates plots from results.
# Author: Arturs Kanepajs, 2025-02-11

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def load_data():
    """Load data from the CSV file"""
    try:
        # First try loading from local file
        file_path = 'Data.csv'
        data_df = pd.read_csv(file_path)
    except FileNotFoundError:
        try:
            # If local file not found, try loading from URL
            url = 'https://raw.githubusercontent.com/akanepajs/multilingual/main/Data.csv'
            data_df = pd.read_csv(url)
        except Exception as e:
            raise Exception(f"Could not load data from file or URL: {str(e)}")
    
    # Append the log column
    data_df['Log CommonCrawl corpus (%)'] = np.log(data_df['CommonCrawl corpus (%)'])
    return data_df

def create_regression_results(data_df):
    """Create regression results DataFrames"""
    # Initialize storage structures
    results_dict = {
        'Model': [], 'Dependent Variable': [], 
        'Intercept': [], 'Slope': [], 'p-value': []
    }
    logit_results_dict = {
        'Model': [], 'Dependent Variable': [], 
        'Intercept': [], 'Slope': [], 'p-value': []
    }
    
    n_total = 100  # observations per country per prompt type
    models_list = sorted(data_df['Model'].unique())
    
    for model_name in models_list:
        model_df = data_df[data_df['Model'] == model_name]
        
        # OLS Regressions
        for dep_var in ['Harmful ACCEPTED', 'Harmless REJECTED']:
            X = sm.add_constant(model_df['Log CommonCrawl corpus (%)'])
            y = model_df[dep_var]
            ols_model = sm.OLS(y, X).fit()
            
            results_dict['Model'].append(model_name)
            results_dict['Dependent Variable'].append(dep_var)
            results_dict['Intercept'].append(ols_model.params['const'])
            results_dict['Slope'].append(ols_model.params['Log CommonCrawl corpus (%)'])
            results_dict['p-value'].append(ols_model.pvalues['Log CommonCrawl corpus (%)'])
        
        # Logistic Regressions
        for dep_var, outcome_type in [('Harmful ACCEPTED', 'Harmful'), ('Harmless REJECTED', 'Harmless')]:
            # Expand data for logistic regression
            expanded_data = []
            for _, row in model_df.iterrows():
                # Positive cases
                expanded_data.extend([(row['Log CommonCrawl corpus (%)'], 1)] * row[f'{outcome_type} {dep_var.split()[-1]}'])
                # Negative cases
                other_outcomes = [x for x in ['ACCEPTED', 'REJECTED', 'UNCLEAR'] 
                                if x != dep_var.split()[-1]]
                for outcome in other_outcomes:
                    expanded_data.extend([(row['Log CommonCrawl corpus (%)'], 0)] * row[f'{outcome_type} {outcome}'])
            
            expanded_df = pd.DataFrame(expanded_data, columns=['Log CommonCrawl corpus (%)', 'Outcome'])
            X_logit = sm.add_constant(expanded_df['Log CommonCrawl corpus (%)'])
            y_logit = expanded_df['Outcome']
            
            logit_model = sm.Logit(y_logit, X_logit).fit(disp=0)
            
            logit_results_dict['Model'].append(model_name)
            logit_results_dict['Dependent Variable'].append(dep_var)
            logit_results_dict['Intercept'].append(logit_model.params['const'])
            logit_results_dict['Slope'].append(logit_model.params['Log CommonCrawl corpus (%)'])
            logit_results_dict['p-value'].append(logit_model.pvalues['Log CommonCrawl corpus (%)'])
    
    return pd.DataFrame(results_dict), pd.DataFrame(logit_results_dict)

def create_and_save_plot(data_df, regression_results_df, logit_results_df, model_name, dep_var):
    """Create and save a plot for a specific model and dependent variable"""
    # Filter data for the selected model
    model_df = data_df[data_df['Model'] == model_name]
    n_total = 100
    
    # Get regression results
    ols_results = regression_results_df[
        (regression_results_df['Model'] == model_name) & 
        (regression_results_df['Dependent Variable'] == dep_var)
    ].iloc[0]
    
    logit_results = logit_results_df[
        (logit_results_df['Model'] == model_name) & 
        (logit_results_df['Dependent Variable'] == dep_var)
    ].iloc[0]
    
    # Calculate regression lines
    sorted_commoncrawl_corpus = np.sort(model_df['CommonCrawl corpus (%)'])
    ols_line = ols_results['Intercept'] + ols_results['Slope'] * np.log(sorted_commoncrawl_corpus)
    logit_curve = 1 / (1 + np.exp(-(logit_results['Intercept'] + 
                                   logit_results['Slope'] * np.log(sorted_commoncrawl_corpus))))
    
    # Prepare observed data
    observed_proportions = model_df[dep_var] / n_total
    lower_bounds, upper_bounds = proportion_confint(model_df[dep_var], n_total, alpha=0.05, method='wilson')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot error bars
    plt.errorbar(model_df['CommonCrawl corpus (%)'], observed_proportions * 100,
                yerr=[(observed_proportions - lower_bounds) * 100, (upper_bounds - observed_proportions) * 100],
                fmt='o', color='blue', ecolor='lightgray', elinewidth=2, capsize=3,
                label='Observed proportion with 95% CI (Wilson score interval)')
    
    # Plot regression lines with fixed syntax
    intercept = logit_results["Intercept"]
    slope = logit_results["Slope"]
    logit_formula = f'P = 1/(1 + e^(-({intercept:.2f} + {slope:.2f}log(x))))'
    plt.plot(sorted_commoncrawl_corpus, logit_curve * 100, color='red',
             label=f'Logistic regression, 2400 obs.: {logit_formula}\nslope coefficient p-value = {logit_results["p-value"]:.3f}')
    
    ols_formula = f'y = {ols_results["Intercept"]:.2f} + {ols_results["Slope"]:.2f}log(x)'
    plt.plot(sorted_commoncrawl_corpus, ols_line, color='green', linestyle='--',
             label=f'OLS regression, 24 obs.: {ols_formula}\nslope coefficient p-value = {ols_results["p-value"]:.3f}')
    
    # Add language labels
    for i, txt in enumerate(model_df['Language_short']):
        plt.annotate(txt, (model_df['CommonCrawl corpus (%)'].iloc[i], observed_proportions.iloc[i] * 100),
                     textcoords="offset points", xytext=(-7, 0), ha='right', fontsize=12)
    
    # Customize plot
    plt.xscale('log')
    plt.xlabel('CommonCrawl Corpus (%), log scale')
    plt.ylabel('%')
    plt.xticks([0.01, 0.1, 1, 10], ['0.01%', '0.1%', '1%', '10%'])
    plt.grid(True, color='gray', alpha=0.2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout()
    
    # Save the plot
    filename = f'{model_name.replace(" ", "_")}_{dep_var.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def main():
    # Load data
    print("Loading data...")
    data_df = load_data()
    print("Data loaded successfully!")
    
    # Create regression results
    print("Performing regressions...")
    regression_results_df, logit_results_df = create_regression_results(data_df)
    print("Regressions completed!")
    
    # Generate plots for all models and dependent variables
    print("\nGenerating plots...")
    models = sorted(data_df['Model'].unique())
    dep_vars = ['Harmful ACCEPTED', 'Harmless REJECTED']
    
    for model in models:
        for dep_var in dep_vars:
            filename = create_and_save_plot(data_df, regression_results_df, logit_results_df, model, dep_var)
            print(f'Generated: {filename}')
    
    print("\nDone! All plots have been generated.")

if __name__ == "__main__":
    main()
