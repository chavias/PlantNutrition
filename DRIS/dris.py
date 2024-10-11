import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from dataclasses import dataclass


def calculate_ratios(df):
    ratios = {}
    for col1, col2 in combinations(df.columns, 2):
        ratio_name1 = f"{col1}/{col2}"
        # ratio_name2 = f"{col2}/{col1}"
        ratios[ratio_name1] = df[col1] / df[col2]
        # ratios[ratio_name2] = df[col2] / df[col1]
    return pd.DataFrame(ratios)


def calculate_CV(df_ratios):
    '''Calculates the coefficient of variation'''
    cv_dict = {}
    for ratio_name, ratio_values in df_ratios.items():
        mean_ratio = ratio_values.mean()
        std_ratio = ratio_values.std()
        cv = (std_ratio / mean_ratio) * 100  # percentage or not 
        cv_dict[ratio_name] = [cv]
    return pd.DataFrame(cv_dict)


def calculate_mean_rations(df_optimum, df_diagnosed):
    df_di = pd.DataFrame(df_diagnosed.mean()).T
    df_diagnosed_mean_ratios = calculate_ratios(df_di)
    df_op = pd.DataFrame(df_optimum.mean()).T
    df_optimum_mean_ratios = calculate_ratios(df_op)
    return df_diagnosed_mean_ratios, df_optimum_mean_ratios


def f(df_diagnosed_mean_ratios, df_optimum_mean_ratios, df_optimum_CV):
    names = df_diagnosed_mean_ratios.columns
    diagnosed_ratios = df_diagnosed_mean_ratios.to_numpy()
    optimum_mean_ratios = df_optimum_mean_ratios.to_numpy()
    optimum_CV = df_optimum_CV.to_numpy()

    def f_single(diagnosed_ratio, optimum_mean_ratio, CV):
        if diagnosed_ratio == 0:
            return float('inf')  # Return inf or some large number to handle zero division
        if diagnosed_ratio >= optimum_mean_ratio:
            f = ((diagnosed_ratio / optimum_mean_ratio) - 1) * 1000 / CV
        else:
            f = (1 - (optimum_mean_ratio / diagnosed_ratio)) * 1000 / CV
        return [f]

    f_dict = dict()
    for i in range(len(names)):
        f_dict[f"f({names[i]})"] = f_single(diagnosed_ratios[0, i], optimum_mean_ratios[0, i], optimum_CV[0, i])
    return pd.DataFrame(f_dict)


def create_index_string(index_element, df_diagnosed):
    elements = df_diagnosed.columns
    result_string = f'I_{{{elements[index_element]}}} ='
    for i, element in enumerate(elements):
        if index_element < i:
            result_string += (f' + f({elements[index_element]}/{element})')
        elif index_element > i:
            result_string += (f' - f({element}/{elements[index_element]})')
    return result_string


def create_all_index_strings(df_diagnosed):
    elements = df_diagnosed.columns
    results_list = []
    for i in range(len(elements)):
        results_list.append(create_index_string(i, df_diagnosed))
    return results_list


def calculate_index_value(index_element, df_diagnosed, df_f):
    f_dict = df_f.to_dict('index')[0]
    elements = df_diagnosed.columns
    result = 0
    for i, element in enumerate(elements):
        if index_element < i:
            result += f_dict[f'f({elements[index_element]}/{element})']
        elif index_element > i:
            result -= f_dict[f'f({element}/{elements[index_element]})']
    return result


def calculate_all_index_values(df_diagnosed, df_f):
    elements = df_diagnosed.columns
    results_dict = dict()
    for i in range(len(elements)):
        results_dict[f'I_{elements[i]}'] = [calculate_index_value(i, df_diagnosed, df_f)]
    return pd.DataFrame(results_dict)



@dataclass
class Results:
    optimum_ratios: pd.DataFrame
    optimum_CV: pd.Series
    optimum_mean_ratios: pd.DataFrame
    diagnosed_ratios: pd.DataFrame
    diagnosed_mean_ratios: pd.DataFrame
    DRIS_equations: list
    f_values: pd.DataFrame
    DRIS_indices: pd.DataFrame


def calculate_DRIS_index(df_diagnosed, df_optimum):
    # calculate ratios
    df_optimum_ratios = calculate_ratios(df_optimum)
    df_diagnosed_ratios = calculate_ratios(df_diagnosed)
    # calculate CV
    df_optimum_CV = calculate_CV(df_optimum_ratios)
    # calculate mean _ratios
    df_op = pd.DataFrame(df_optimum.mean()).T
    df_optimum_mean_ratios = calculate_ratios(df_op)
    df_di = pd.DataFrame(df_diagnosed.mean()).T
    df_diagnosed_mean_ratios = calculate_ratios(df_di)
    # calculate f
    df_f = f(df_diagnosed_mean_ratios, df_optimum_mean_ratios, df_optimum_CV)
    # calculate indices
    DRIS_equations = create_all_index_strings(df_diagnosed)
    DRIS_indices = calculate_all_index_values(df_diagnosed, df_f)


    # store results in dataclass
    result = Results(
        optimum_ratios=df_optimum_ratios,
        diagnosed_ratios=df_diagnosed_ratios,
        optimum_mean_ratios=df_optimum_mean_ratios,
        diagnosed_mean_ratios=df_diagnosed_mean_ratios,
        optimum_CV=df_optimum_CV,
        f_values=df_f,
        DRIS_indices=DRIS_indices,
        DRIS_equations=DRIS_equations
    )
    return result


def export_to_excel(df,file_name = 'dris.xlsx'):
    df.to_excel(file_name, index=False)