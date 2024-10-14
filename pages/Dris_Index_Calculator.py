import streamlit as st
import pandas as pd
import DRIS.dris as dris
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="DRIS Index Calculator", page_icon="📈")

st.markdown("# DRIS Index Calculator")
st.sidebar.header("DRIS Index Calculator")
st.write(
    """This App calculates the DRIS index given a diagnosed and optimal concentration"""
)

# st.title('DRIS Indices Calculator')

# st.markdown('## Diagnosis')

@st.cache_data
def load_data(uploaded_file):
    # Determine the file type based on the extension
    file_extension = uploaded_file.name.split('.')[-1]  # Get the file extension
    match file_extension:
        case "xlsx":
            df = pd.read_excel(uploaded_file)  # Load Excel file
        case "csv":
            df = pd.read_csv(uploaded_file)  # Load CSV file
        case "tsv":
            df = pd.read_csv(uploaded_file, delimiter='\t')  # Load TSV file
        case _:
            st.error("Unsupported file type. Please upload an xlsx, csv, or tsv file.")
            return None  # Return None if unsupported file type
    return df

def create_bar_plot(df):
    row = df.iloc[0]
    # Create a bar plot
    # x = [f'${index}$' for index in row.index]
    fig = go.Figure(
        data=go.Bar(
            x=row.index,      # Categories (column names)
            y=row.values      # Values in the row
        )
    )
    # Update layout to add titles and improve appearance
    fig.update_layout(
        # title='Bar Plot DRIS Indices',
        xaxis_title='Elements',
        yaxis_title='DRIS index',
        template='plotly_white'
    )
    return fig


# File uploader for diagnosed
df_diagnosed_file = st.file_uploader("Choose diagnosis concentration file ...", type=["xlsx", "csv", "tsv"])
df_diagnosed = None
if df_diagnosed_file is not None:
    df_diagnosed = load_data(df_diagnosed_file)
if df_diagnosed is not None and st.checkbox('Show diagnosed input'):
    st.markdown('#### Diagnosed Input')
    st.write(df_diagnosed)
    st.markdown('#### Diagnosed Input Statistic')
    st.write(df_diagnosed.describe())

# File uploader for optimum
df_optimum_file = st.file_uploader("Choose optimum concentration file...", type=["xlsx", "csv", "tsv"])
df_optimum = None
if df_optimum_file is not None:
    df_optimum = load_data(df_optimum_file)
if df_optimum is not None and st.checkbox('Show optimum input'):
    st.markdown('#### Optimum Input')
    st.write(df_optimum)
    st.markdown('#### Optimum Input Statistic')
    st.write(df_optimum.describe())

# def show_all_index_strings(df_diagnosed, df_f):
#     elements = df_diagnosed.columns
#     for i in range(len(elements)):
#         results_dict[f'I_{elements[i]}'] = [dris.create_index_string(i, df_diagnosed, df_f)]
#     return pd.DataFrame(results_dict)

dris_results = None
if df_optimum_file and df_optimum is not None:
    dris_results = dris.calculate_DRIS_index(df_diagnosed, df_optimum)
    
    # optimum_ratios=df_optimum_ratios,
    # diagnosed_ratios=df_diagnosed_ratios,
    # optimum_mean_ratios=df_optimum_mean_ratios,
    # diagnosed_mean_ratios=df_diagnosed_mean_ratios,
    # optimum_CV=df_optimum_CV,
    # DRIS_indices=DRIS_indices,
    # f_values=df_f,
    # DRIS_indices=DRIS_indices
    


if dris_results is not None: #and st.checkbox('Show DRIS index'):
    st.subheader('Dris Index')
    st.write(dris_results.DRIS_indices)
    fig_excess = create_bar_plot(dris_results.DRIS_indices)
    st.plotly_chart(fig_excess, use_container_width=True)
    if st.checkbox('Show DRIS Equation'):
        equations = dris_results.DRIS_equations
        for i in range(len(equations)):
            st.markdown(f"${equations[i]}$")
    st.subheader('Optimum Coefficient of Variance')
    st.write(dris_results.optimum_CV)