# %% [markdown]
# # Nutrition

# %%
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator#, Akima1DInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def process_data(data_path):
    df = pd.read_excel(data_path)
    time_column = df['Time']
    df_no_time = df.drop(columns='Time')
    X = df_no_time.to_numpy()
    sorted_data = np.sort(X, axis=0)[::-1]
    maxes = np.max(sorted_data, axis=0)
    normalized_data = sorted_data / maxes
    data_df = pd.DataFrame(normalized_data, columns=df_no_time.columns)
    data_df['Time'] = time_column.sort_values(ascending=True).reset_index(drop=True)
    return data_df, maxes


def interpolate_data(data_df, num_points=500, kind='cubic'):
    data = data_df.to_numpy()
    x_original = np.arange(data.shape[0])  # Original index positions
    x_interp = np.linspace(0, data.shape[0] - 1, num_points)  # Interpolation index positions
    
    # Initialize an array to store interpolated data
    Y_interp = np.zeros((num_points, data.shape[1]))
    
    # Interpolate each column
    if kind!='pchip':
        for col in range(data.shape[1]):
            f = interp1d(x_original, data[:, col], kind=kind)  # Use the specified interpolation method
            Y_interp[:, col] = f(x_interp)
    else:
        for col in range(data.shape[1]):
            pchip_interpolator = PchipInterpolator(x_original, data[:, col])  # Create PCHIP interpolator
            Y_interp[:, col] = pchip_interpolator(x_interp)

    # Create a DataFrame for the interpolated data
    interpolated_df = pd.DataFrame(Y_interp, columns=data_df.columns)
    return interpolated_df


def plot_curves_interactive(data_df):
    # Extract 'Time' column and other data
    time = data_df['Time']
    y = data_df.drop(columns='Time')  # Drop 'Time' to get only the data for plotting

    num_cols = y.shape[1]  # Number of columns to plot
    num_rows = (num_cols + 2) // 3  # Calculate number of rows needed for subplots

    # Create subplots figure with plotly
    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=[col for col in y.columns],
                        vertical_spacing=0.15, horizontal_spacing=0.1)

    # Plot each column against 'Time'
    for i, col in enumerate(y.columns):
        row = i // 3 + 1  # Determine row position
        col_position = i % 3 + 1  # Determine column position

        fig.add_trace(
            go.Scatter(
                x=time,
                y=y[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ),
            row=row,
            col=col_position
        )

        # Customize the axis titles and range for each subplot
        fig.update_xaxes(title_text='Days', row=row, col=col_position)
        fig.update_yaxes(title_text='Concentration', row=row, col=col_position, range=[0, None])

    # Update layout for better readability
    fig.update_layout(
        height=300 * num_rows, width=900, title_text="Concentrations Over Time",
        showlegend=False,  # Hide the legend since each subplot has a title
        template='plotly_white'
    )
    
    return fig


# Example usage:
# df_data_interp = df_data_normalized_interp.drop(columns='Time')*maxes
# df_data_interp['Time'] = df_data_normalized_interp['Time']
# plot_curves_interactive(df_data_interp)


def absolute_plant_uptake_during_interval(data_df, start, end):
    # Extract the 'Time' column
    time = data_df['Time'].to_numpy()
    
    # Find the nearest indices for the start and end times
    start_index = np.argmin(np.abs(time - start))
    end_index = np.argmin(np.abs(time - end))
    
    # Calculate uptake as the negative of the nutrient change in solution
    uptake = -(data_df.iloc[end_index, 0:-1] - data_df.iloc[start_index, 0:-1])  # Skip 'Time' column

    return uptake


def nutrients(param, fertilizer, target):
    parameter = np.array(param).reshape(fertilizer.shape[0],1)
    weighted_fertilizers = np.multiply(fertilizer,parameter)
    total_fertilizers = weighted_fertilizers.sum(axis=0)
    value = target - total_fertilizers
    return value


def objective_function(param, fertilizer, target):
    parameter = np.array(param).reshape(fertilizer.shape[0],1)
    # print(f"{parameter.shape=}")
    # print(f"{fertilizer.shape=}")
    weighted_fertilizers = np.multiply(fertilizer,parameter)
    # print(f"{weighted_fertilizers.shape}")
    total_fertilizers = weighted_fertilizers.sum(axis=0)
    # print(f"{total_fertilizers.shape}")
    value = target - total_fertilizers
    return np.sum(np.abs(value))


def constraint_function(param, fertilizer, target):
    # Reshape the parameter vector and compute weighted fertilizers
    parameter = np.array(param).reshape(fertilizer.shape[0], 1)
    weighted_fertilizers = np.multiply(fertilizer, parameter)
    # Sum the weighted fertilizers to get the total contribution for each nutrient
    total_fertilizers = weighted_fertilizers.sum(axis=0)
    # Calculate the difference between the target and total fertilizers for each nutrient
    value = target - total_fertilizers
    # Return the difference (this will ensure SLSQP enforces the constraint for each nutrient)
    return value


def optimize(fertilizer, target, constraint_function):
    # Initial guess for the parameters, one for each row (fertilizer) in the fertilizer DataFrame
    initial_guess = np.zeros(fertilizer.shape[0])
    
    # Define bounds for each parameter (non-negative values)
    bounds = [(0, None) for _ in range(fertilizer.shape[0])]
    
    # Define the constraint dictionary using the provided constraint function
    constraints = {'type': 'ineq', 'fun': constraint_function, 'args': (fertilizer, target)}
    
    # Call the optimizer
    result = minimize(
        objective_function,
        initial_guess,
        args=(fertilizer, target),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )
    return result  # Return optimized parameters and minimized objective value


def calculate_nutrient_schedule(path_data, path_fertilizer, time_intervall_days):
    # Load data
    df_data_normalized, maxes = process_data(path_data)
    df_fertilizer_normalized = pd.read_excel(path_fertilizer)/maxes

    # Interpolate data to a timestep of days if necessary
    num_days = df_data_normalized['Time'].max()
    df_data_normalized_interp = interpolate_data(df_data_normalized, num_points=num_days, kind='pchip')
    
    # Number of fertilizing days
    num_fertilization_events = num_days // time_intervall_days

    results = dict()
    left_over = 0
    for i in range(num_fertilization_events):
        # uptake of the plant during the time intervall
        start = i*time_intervall_days
        end = (i+1)*time_intervall_days
        uptake = absolute_plant_uptake_during_interval(df_data_normalized_interp, start, end)
        target_concentration = np.abs(uptake.to_numpy() - left_over) # substract left over nutrients
        # optimize fertilizer amounts
        result = optimize(df_fertilizer_normalized,
                          target_concentration,
                          constraint_function)
        # calculate left over
        result.left_over = nutrients(result.x, df_fertilizer_normalized, target_concentration)
        # store result
        results[f"{int(end)}"] = result
        
    return results, maxes


def plot_stacked_interactive(df, fig_title, y_title, legend_title):
    # Get element names (these are your columns)
    element_columns = df.columns

    # Prepare the x-axis labels (row index keys from the DataFrame)
    index_labels = df.index

    # Initialize the figure
    fig = go.Figure()

    # Initialize the bottom of the stack (start with zeros)
    bottom = np.zeros(len(index_labels))

    # Add each element as a trace
    for col in element_columns:
        element_values = df[col].values
        rounded_values = np.round(element_values, 3)  # Round to 3 decimal places

        # Add a bar trace for each element
        fig.add_trace(go.Bar(
            x=index_labels,
            y=element_values,
            name=col,
            text=[f'{val:.3f}' for val in rounded_values],  # Format values to 3 decimal places
            hoverinfo='text+name',  # Show value and label name on hover
            offsetgroup=0,
            base=bottom
        ))

        # Update the bottom position for the next stack
        bottom += element_values

    # Update layout for better readability and display all x-axis labels
    fig.update_layout(
        title=fig_title,
        xaxis=dict(
            title='Day',
            type='category',                  # Use categorical axis for clear labeling
        ),
        yaxis=dict(title=y_title),
        barmode='stack',
        legend=dict(title=legend_title),
        template='plotly_white',
        hovermode="x"
    )
    return fig


def create_left_over_df(optimization_results,maxes):
    data = {}
    for k, result in optimization_results.items():
        data[k] = result.left_over*maxes
    df = pd.DataFrame(data).T  # Transpose to have keys as row indexes
    return df


def create_fertilizer_df(optimization_results,time_step):
    data = {}
    for k, result in optimization_results.items():
        data[str(int(k)-time_step)] = result.x
    df = pd.DataFrame(data).T  # Transpose to have keys as row indexes
    return df
