import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import streamlit as st

def process_data_df(df):
    if 'Time' in df.columns:
        df_time = df['Time']
        df_no_time = df.drop(columns='Time')
    elif 'time' in df.columns:
        df_time = df['time']
        df_no_time = df.drop(columns='time')
    elif 'days' in df.columns:
        df_time = df['days']
        df_no_time = df.drop(columns='days')
    elif 'Days' in df.columns:
        df_no_time = df.drop(columns='Days')
        df_time = df['Days']
    else:
        st.error("Error: None of the columns 'Time', 'time', 'Days' or 'days' were found.")
        # st.stop()
    #time = df_time.to_numpy()
    X = df_no_time.to_numpy()
    # Sort data along axis=0 and reverse to get descending order
    data = np.sort(X, axis=0)[::-1]
    # normalize data to 1
    maxes = np.max(data, axis=0)
    data = data / maxes
    return data, maxes


def interpolate_data(data, num_points=500, kind='cubic'):
    x_original = np.arange(data.shape[0])  # Original index positions
    x_interp = np.linspace(0, data.shape[0] - 1, num_points)  # Interpolation index positions
    # Initialize an array to store interpolated data
    Y_interp = np.zeros((num_points, data.shape[1]))
    # Interpolate each column
    for col in range(data.shape[1]):
        f = interp1d(x_original, data[:, col], kind='cubic')  # Use cubic interpolation for smoothness
        Y_interp[:, col] = f(x_interp)
    return x_interp, Y_interp


def absolute_plant_uptake_during_interval(x_interp, Y_interp, start, end):
    start_index = np.argmin(np.abs(x_interp - start)) # find the best index
    end_index = np.argmin(np.abs(x_interp - end))
    # uptake is the negative of nutrient change in solution
    return -(Y_interp[end_index,:] - Y_interp[start_index,:]) 


def nutrients(params, c_fertilizer1, c_fertilizer2, c_fertilizer3, c_target):
    a, b, c = params
    value = c_target - a*c_fertilizer1 - b*c_fertilizer2 - c*c_fertilizer3
    return value  


def objective_function(params, c_fertilizer1, c_fertilizer2, c_fertilizer3, c_target):
    a, b, c = params
    value = np.sum(np.abs(c_target - a*c_fertilizer1 - b*c_fertilizer2 - c*c_fertilizer3))
    return value  


def constraint_function(params, c_fertilizer1, c_fertilizer2, c_fertilizer3, c_target):
    a, b, c = params
    return np.min(c_target - a*c_fertilizer1 - b*c_fertilizer2 - c*c_fertilizer3)


def optimize(c_fertilizer1, c_fertilizer2, c_fertilizer3, c_target, constraint):
    # Initial guess for a and b
    initial_guess = [0, 0, 0]
    bounds = [(0, None), (0, None), (0, None)]  # (lower bound, upper bound) for a, b and c
    # Call the optimizer
    result = minimize(objective_function, initial_guess, 
                      args=(c_fertilizer1, c_fertilizer2, c_fertilizer3, c_target),
                      bounds = bounds,  # (lower bound, upper bound) for a and b
                      constraints=[constraint], 
                      method='SLSQP') 
    
    # Extract the optimized values of a and b
    a_opt, b_opt, c_opt = result.x
    return a_opt, b_opt, c_opt, result.fun  # Return optimized a, b, and the minimized value


def calculate_fertilization_schedule_df(df, time_intervall_days, c_fertilizer1, c_fertilizer2, c_fertilizer3):
    # read and normalize data
    data, maxes = process_data_df(df=df)
    # normalize fartilizer
    norm_c_fertilizer1 = c_fertilizer1/maxes
    norm_c_fertilizer2 = c_fertilizer2/maxes
    norm_c_fertilizer3 = c_fertilizer3/maxes

    # print(f"{norm_c_fertilizer1}")
    # interpolate the data
    x_interp, Y_interp = interpolate_data(data, num_points=4*7, kind='linear')
    # calculate the uptake of each nutrient of the plant
    # x_interp, plant_uptake_rate = calculate_nutrient_uptake_rate(x_interp, Y_interp, num_interp_points=500)
    # calculate 
    time_intervall_week = time_intervall_days/7
    num_fertilization_events = int(x_interp[-1] // time_intervall_week)
    # allocate result dictionary
    results = dict()
    # build schedule
    residual = 0
    for i in range(num_fertilization_events):
        # define start and end times
        start = i*time_intervall_week
        end = (i+1)*time_intervall_week
        # calculate target without taking into account residuals
        c_target_without_residuals = absolute_plant_uptake_during_interval(x_interp=x_interp, Y_interp=Y_interp, start=start ,end=end)
        # calculate the target considering residual (excess) nutrients in the solution
        c_target = np.abs(c_target_without_residuals - residual) # prevent target being negative
        # Define the constraint
        constraint = {
            'type': 'ineq',  # 'ineq' means the constraint function must return >= 0
            'fun': constraint_function,
            'args': (norm_c_fertilizer1, norm_c_fertilizer2, norm_c_fertilizer3, c_target)
        }
        # optimize 
        a_opt, b_opt, c_opt, minimized_value = optimize(norm_c_fertilizer1, norm_c_fertilizer2, norm_c_fertilizer3, c_target, constraint)
        params_opt = (a_opt, b_opt, c_opt)
        # calculate the excess nutrient after time period
        residual = nutrients(params_opt, norm_c_fertilizer1, norm_c_fertilizer2, norm_c_fertilizer3, c_target)
        # store the results
        results[f"{int(start*7)}"] = {'f1' : a_opt,
                                     'f2' : b_opt,
                                     'f3' : c_opt,
                                    #  'excess' : (residual*maxes).round(3),
                                    #  'excess_normalized' : (residual).round(3),
                                     'excess_1': residual[0],
                                     'excess_2': residual[1],
                                     'excess_3': residual[2],
                                     'excess_4': residual[3],
                                     'excess_5': residual[4],
                                     'excess_6': residual[5],
                                     'excess_7': residual[6],
                                     'excess_8': residual[7],
                                     'excess_9': residual[8],
                                     'excess_10': residual[9],
                                     'excess_11': residual[10],
                                     'objective_function': minimized_value}
    return pd.DataFrame(results).T


def plot_excess_stacked_interactive_streamlit(df_transposed):
    # Identify all columns that start with 'excess_'
    excess_columns = [col for col in df_transposed.columns if col.startswith('excess_')]

    # Prepare the x-axis labels
    days = df_transposed.index
    day_array = [int(day) for day in days]
    x_labels = [day + day_array[1] for day in day_array]

    # Initialize the figure
    fig = go.Figure()

    # Initialize the bottom of the stack
    bottom = np.zeros(len(days))

    # Add each excess group as a trace
    for col in excess_columns:
        # Flatten or aggregate the column values if they are lists or arrays
        excess_values = df_transposed[col].apply(lambda x: np.sum(x) if isinstance(x, (list, np.ndarray)) else x).values
        rounded_values = np.round(excess_values, 3)  # Round to 3 decimal places

        # Add a bar trace for each excess group
        fig.add_trace(go.Bar(
            x=x_labels,
            y=excess_values,
            name=col,
            text=[f'{val:.3f}' for val in rounded_values],  # Format values to 3 decimal places
            hoverinfo='text+name',  # Show value and label name on hover
            offsetgroup=0,
            base=bottom
        ))
        
        # Update the bottom position for the next stack
        bottom += excess_values

    # Update layout for better readability and display all x-axis labels
    fig.update_layout(
        title='Stacked Excess Values for Different Days',
        xaxis=dict(
            title='Days',
            tickmode='linear',                # Force linear tick mode
            tickvals=x_labels,                # Set tick values to match x-axis labels
            ticktext=x_labels,                # Ensure each bar has a label
            type='category',                  # Force categorical x-axis to display all unique values
            # categoryorder='category ascending'  # Order categories in ascending order to prevent skipping
        ),
        yaxis=dict(title='Excess Value'),
        barmode='stack',
        legend=dict(title='Excess Groups'),
        template='plotly_white',
        hovermode="x"
    )
    return fig