import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path
from scipy.stats import chi2_contingency, f_oneway, jarque_bera, ttest_ind, levene, mannwhitneyu
from matplotlib.gridspec import GridSpec
from src.constants.constants import DATA_DIR


def get_data(name: str, data_type_dir: str) -> pd.DataFrame:
    """
    Reads a CSV file from the specified data directory.

    Args:
        name (str): Name of the file (without extension).
        data_type_dir (str): Subdirectory type (e.g., 'raw', 'processed').

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    file_name = f"{name}.csv"
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    file_path = Path(root_path) / DATA_DIR / data_type_dir / file_name
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    

def outlier_overview(dataframe: pd.DataFrame, column_name: str, cat_col: Optional[str] = None) -> pd.DataFrame:
    """
    Provides an overview of a numerical column by plotting a box plot and returning descriptive statistics.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to analyze.
        cat_col (Optional[str]): The name of the categorical column for grouping (optional).
    
    Returns:
        pd.DataFrame: Descriptive statistics of the numerical column. If `cat_col` is provided, 
        statistics are grouped by the categorical column.
    
    Notes:
        - The function generates a box plot for visualizing outliers and spread.
        - If `cat_col` is specified, the box plot compares distributions across categories.
    """
    # Calculate descriptive statistics
    stats = (
        dataframe.groupby(cat_col)[column_name].describe()
        if cat_col
        else dataframe[column_name].describe()
    )

    # Create the box plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=dataframe, x=cat_col, y=column_name, ax=plt.gca())
    plt.title(f"Box Plot of '{column_name}'" + (f" by '{cat_col}'" if cat_col else ""))
    plt.tight_layout()
    plt.show()

    return stats


def export_data(dataframe: pd.DataFrame, name: str) -> None:
    """
    Exports a DataFrame to a CSV file in the 'data/export' directory.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be exported.
        name (str): Name of the file (without extension).

    Returns:
        None
    """

    # Define the export directory path
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    export_dir = root_path / DATA_DIR / "export"
    export_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Construct the full file path
    file_path = export_dir / f"{name}.csv"

    try:
        # Save the DataFrame as a CSV
        dataframe.to_csv(file_path, index=False)
        print(f"Data successfully exported to: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to export data to {file_path}") from e


def prepare_segment_trends_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset for analyzing segment trends and insights by:
    1. Filling missing values in the 'extended_upto' column with 0 where 'extended_memory_available' is 0.
    2. Removing smartphones with a price greater than ₹2,00,000.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame ready for analysis.
    """
    return (
        dataframe
        # Fill missing values in 'extended_upto' where 'extended_memory_available' is 0
        .assign(extended_upto=lambda x: x['extended_upto'].fillna(0).where(x['extended_memory_available'] == 0, x['extended_upto']))
        # Remove smartphones with price > 200000
        .loc[lambda x: x['price'] <= 200000]
    )


def numerical_analysis(dataframe: pd.DataFrame, column_name: str, cat_col: Optional[str] = None, bins: str = "auto") -> None:
    """
    Perform numerical analysis with KDE, boxplot, and histogram.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Column name for numerical analysis.
        cat_col (Optional[str]): Categorical column for grouping (hue). Default is None.
        bins (str): Number of bins or binning strategy for histogram. Default is "auto".

    Returns:
        None
    """
    fig = plt.figure(figsize=(15, 10))
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])
    sns.kdeplot(data=dataframe, x=column_name, hue=cat_col, ax=ax1)
    sns.boxplot(data=dataframe, x=column_name, hue=cat_col, ax=ax2)
    sns.histplot(data=dataframe, x=column_name, bins=bins, hue=cat_col, kde=True, ax=ax3)
    plt.tight_layout()
    plt.show()


def numerical_categorical_analysis(dataframe: pd.DataFrame, cat_column_1: str, num_column: str) -> None:
    """
    Perform numerical-categorical analysis with barplot, boxplot, violin plot, and strip plot.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        cat_column_1 (str): Categorical column for x-axis.
        num_column (str): Numerical column for y-axis.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 7.5))
    sns.barplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[0])
    sns.boxplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[1])
    sns.violinplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[0])
    sns.stripplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[1])
    plt.tight_layout()
    plt.show()


def categorical_analysis(dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Perform categorical analysis with value counts and countplot.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Column name for analysis.

    Returns:
        None
    """
    display( # type: ignore
        pd.DataFrame({
            "Count": dataframe[column_name].value_counts(),
            "Percentage": dataframe[column_name].value_counts(normalize=True)
            .mul(100)
            .round(2)
            .astype("str")
            .add("%")
        })
    )
    print("*" * 50)
    unique_categories = dataframe[column_name].unique().tolist()
    number_of_categories = dataframe[column_name].nunique()
    print(f"The unique categories in {column_name} column are {unique_categories}")
    print("*" * 50)
    print(f"The number of categories in {column_name} column are {number_of_categories}")
    sns.countplot(data=dataframe, x=column_name)
    plt.xticks(rotation=45)
    plt.show()


def multivariate_analysis(dataframe: pd.DataFrame, num_column: str, cat_column_1: str, cat_column_2: str) -> None:
    """
    Perform multivariate analysis with barplot, boxplot, violin plot, and strip plot.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        num_column (str): Numerical column for y-axis.
        cat_column_1 (str): Primary categorical column for x-axis.
        cat_column_2 (str): Secondary categorical column for hue.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 7.5))
    sns.barplot(data=dataframe, x=cat_column_1, y=num_column, hue=cat_column_2, ax=ax1[0])
    sns.boxplot(data=dataframe, x=cat_column_1, y=num_column, hue=cat_column_2, ax=ax1[1])
    sns.violinplot(data=dataframe, x=cat_column_1, y=num_column, hue=cat_column_2, ax=ax2[0])
    sns.stripplot(data=dataframe, x=cat_column_1, y=num_column, hue=cat_column_2, dodge=True, ax=ax2[1])
    plt.tight_layout()
    plt.show()


def chi_2_test(dataframe: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> None:
    """
    Perform Chi-squared test for independence between two categorical variables.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        col1 (str): First categorical column.
        col2 (str): Second categorical column.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        None
    """
    data = dataframe.loc[:, [col1, col2]].dropna()
    contingency_table = pd.crosstab(data[col1], data[col2])
    _, p_val, _, _ = chi2_contingency(contingency_table)
    print(p_val)
    if p_val <= alpha:
        print(f"Reject the null hypothesis. There is a significant association between {col1} and {col2}.")
    else:
        print(f"Fail to reject the null hypothesis. There is no significant association between {col1} and {col2}.")


def anova_test(dataframe: pd.DataFrame, num_col: str, cat_col: str, alpha: float = 0.05) -> None:
    """
    Perform ANOVA test to determine the relationship between a numerical and categorical variable.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        num_col (str): Numerical column.
        cat_col (str): Categorical column (can include boolean types).
        alpha (float): Significance level. Default is 0.05.

    Returns:
        None
    """
    data = dataframe.loc[:, [num_col, cat_col]].dropna()
    
    # Ensure the categorical column is converted to a categorical type
    if data[cat_col].dtype == 'bool':
        data[cat_col] = data[cat_col].astype('category')
    
    cat_group = data.groupby(cat_col, observed=False)
    groups = [group[num_col].values for _, group in cat_group]
    
    # Perform the ANOVA test
    _, p_val = f_oneway(*groups)
    
    # Interpret the p-value
    print(f"ANOVA p-value: {p_val}")
    if p_val <= alpha:
        print(f"Reject the null hypothesis. There is a significant relationship between {num_col} and {cat_col}.")
    else:
        print(f"Fail to reject the null hypothesis. There is no significant relationship between {num_col} and {cat_col}.")


def test_for_normality(dataframe: pd.DataFrame, column_name: str, alpha: float = 0.05) -> None:
    """
    Test for normality using Jarque-Bera test.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Column name to test for normality.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        None
    """
    data = dataframe[column_name]
    print("Jarque Bera Test for Normality")
    _, p_val = jarque_bera(data)
    print(p_val)
    if p_val <= alpha:
        print(f"Reject the null hypothesis. The data is not normally distributed.")
    else:
        print(f"Fail to reject the null hypothesis. The data is normally distributed.")


def two_sample_independent_ttest(dataframe: pd.DataFrame, num_col: str, cat_col: str, alpha: float = 0.05) -> None:
    """
    Perform a two-sample independent t-test for a numerical variable across two categories of a categorical variable.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        num_col (str): Numerical column.
        cat_col (str): Categorical column with two categories.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        None
    """
    # Drop NaNs
    data = dataframe.loc[:, [num_col, cat_col]].dropna()
    
    # Ensure the categorical column has only two categories
    categories = data[cat_col].unique()
    if len(categories) != 2:
        raise ValueError(f"{cat_col} must have exactly two unique categories, but found {len(categories)}.")
    
    # Split numerical data into two groups based on the categories
    group1 = data[data[cat_col] == categories[0]][num_col]
    group2 = data[data[cat_col] == categories[1]][num_col]
    
    # Check for equal variances using Levene's test
    _, p_var = levene(group1, group2)
    equal_var = p_var > alpha
    
    # Perform two-sample t-test
    t_stat, p_val = ttest_ind(group1, group2, equal_var=equal_var)
    
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    if p_val <= alpha:
        print(f"Reject the null hypothesis. The means of {num_col} differ between the two categories of {cat_col}.")
    else:
        print(f"Fail to reject the null hypothesis. The means of {num_col} do not differ between the two categories of {cat_col}.")


def mann_whitney_test(
    dataframe: pd.DataFrame, 
    num_col: str, 
    cat_col: str, 
    group_1: any, 
    group_2: any, 
    alpha: float = 0.05
) -> None:
    """
    Perform the Mann-Whitney U Test to compare the distribution of a numerical variable 
    across two groups in a categorical variable.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        num_col (str): Numerical column.
        cat_col (str): Categorical column with two unique categories.
        group_1 (any): The first category in the categorical column.
        group_2 (any): The second category in the categorical column.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        None: Prints the test statistic and p-value, and interprets the result.
    """
    # Validate unique categories in the categorical column
    unique_categories = dataframe[cat_col].dropna().unique()
    if len(unique_categories) != 2:
        raise ValueError(f"The categorical column '{cat_col}' must have exactly two unique values. Found: {unique_categories}")
    
    # Split data into two groups
    group_1_data = dataframe[dataframe[cat_col] == group_1][num_col].dropna()
    group_2_data = dataframe[dataframe[cat_col] == group_2][num_col].dropna()
    
    # Perform Mann-Whitney U Test
    stat, p_value = mannwhitneyu(group_1_data, group_2_data, alternative='two-sided')
    
    # Display results
    print(f"Mann-Whitney U Test Results for {num_col} by {cat_col}:")
    print(f"Test Statistic: {stat}")
    print(f"P-Value: {p_value}")
    if p_value <= alpha:
        print(f"Reject the null hypothesis: There is a significant difference in {num_col} between {group_1} and {group_2}.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference in {num_col} between {group_1} and {group_2}.")


def levene_test(dataframe: pd.DataFrame, num_col: str, cat_col: str) -> None:
    """
    Perform Levene's Test to check for homogeneity of variances across different groups in a categorical column.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        num_col (str): Numerical column to test for variance equality.
        cat_col (str): Categorical column representing the groups to compare.

    Returns:
        None
    """
    # Drop NaN values
    data = dataframe.loc[:, [num_col, cat_col]].dropna()

    # Group the data by the categorical column and create a list of values for each group
    groups = [group[num_col].values for _, group in data.groupby(cat_col)]
    
    # Perform Levene's Test
    stat, p_val = levene(*groups)
    
    print(f"Levene's Test p-value: {p_val}")
    
    if p_val <= 0.05:
        print(f"Reject the null hypothesis. The variances across the groups are significantly different.")
    else:
        print(f"Fail to reject the null hypothesis. The variances across the groups are not significantly different.")
