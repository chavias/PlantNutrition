import numpy as np
import pandas as pd


def calculate_z(df):
    ''' Calculates z for CND analysis on a DataFrame input.
      Args:
        df (pd.DataFrame): DataFrame with columns representing nutrient concentrations for each plant.
      Returns:
        pd.DataFrame: DataFrame containing z values with column names prefixed by "z_".
    '''
    # Normalize each row so that the sum of nutrients is 1
    row_sums = df.sum(axis=1)
    x = df.div(row_sums, axis=0)

    # Calculate the geometric mean for each row
    g = x.prod(axis=1)**(1/x.shape[1])

    # Compute z values
    z = np.log(x.div(g, axis=0))

    # Rename columns to reflect that they are z values
    z.columns = [f'z_{col}' for col in df.columns]

    return z


def calculate_cnd_index(df_diagnosed, df_optimum):
    ''' 
    Calculates I for CND analysis using DataFrame inputs, based on the mean z values.
      Args:
        df_diagnosed (pd.DataFrame): DataFrame with nutrient concentrations of the diagnosed population.
        df_optimum (pd.DataFrame): DataFrame with nutrient concentrations of the optimum (target) population.
      Returns:
        pd.DataFrame: DataFrame containing I values with column names prefixed by "I_".
    '''
    # Calculate z values for diagnosed and optimum using the calculate_z function
    z_diagnosed = calculate_z(df_diagnosed)
    z_optimum = calculate_z(df_optimum)

    # Calculate mean z values for each nutrient
    mean_z_diagnosed = z_diagnosed.mean(axis=0)
    mean_z_optimum = z_optimum.mean(axis=0)

    # Calculate standard deviations for each nutrient across optimum rows
    stds = z_optimum.std(axis=0)

    # Calculate I values for each nutrient using the mean z values
    I_values = (mean_z_diagnosed.values - mean_z_optimum.values) / stds.values

    # Convert the I values to a DataFrame and rename columns to reflect they are I values
    I_values_df = pd.DataFrame([I_values], columns=[f'I_{col}' for col in df_diagnosed.columns])

    return I_values_df
