import streamlit as st
import pandas as pd
import plant_scheduler.nutrient_schedule_class as ps
import numpy as np

# st.title('Nutrient schedule')
st.set_page_config(page_title="Nutrient Schedule", page_icon="🧪")

st.markdown("# 🧪 Nutrient schedule")
st.sidebar.header("Nutrient schedule")
st.write(
    """Calculate the optimal nutrient schedule given a optimal nutrient concentration time series and fertilizer."""
)


# st.markdown('## Nutrient Concentration')
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


st.markdown("## Upload Files")
# File uploader for the nutrient concentration
uploaded_concentration_file = st.file_uploader("Choose file of optimal nutrient time series ...", type=["xlsx", "csv", "tsv"])
data = None  # Initialize data

if uploaded_concentration_file is not None:
    data = load_data(uploaded_concentration_file)

if data is not None:
    if st.checkbox('Show nutrient concentration data'):
        st.subheader('Nutrient concentration')
        st.write(data)
        raw_curves = ps.plot_curves_interactive(data)
        st.plotly_chart(raw_curves)
    if st.checkbox('Show nutrient concentration interpolated'):
        interpol = ps.calc_interpolate_data(data)
        st.write(interpol)
        interp_curve = ps.plot_curves_interactive(interpol)
        st.plotly_chart(interp_curve)

uploaded_fertilizer_file = st.file_uploader("Choose file of nutrient concentration of fertilizer ...", type=["xlsx", "csv", "tsv"])
fertilizer = None  # Initialize data

if uploaded_fertilizer_file is not None:
    fertilizer = load_data(uploaded_fertilizer_file)

if fertilizer is not None and st.checkbox('Show fertilizer concentration data'):
    st.subheader('fertilizer concentration')
    st.write(fertilizer)


if uploaded_concentration_file is not None and uploaded_fertilizer_file is not None:
    no = ps.NutrientOptimization(data_path=uploaded_concentration_file,
                                 fertilizer_path=uploaded_fertilizer_file)
    
    # if st.checkbox('Show interpolated data'):
    #     interp_curves = ps.plot_curves_interactive(no.df_data_normalized_interp)
    #     st.plotly_chart(interp_curves)
    
    st.markdown("## Create Nutrient Schedule")
    time_intervall_days = st.selectbox(
    "Time intervall for fertilizing?",
    list(range(1, data['Time'].max()+1)),
    )
    # st.write(f"You selected: {time_intervall_days} days")

    no.time_intervall = time_intervall_days

    results = no.calculate_nutrient_schedule()
    
    df_fertilizer = no.create_fertilizer_df(results)
    df_left_over = no.create_left_over_df(results)

    # nutrient excess
    fig_left_over = ps.plot_stacked_interactive(df_left_over,
                                                fig_title='Nutrient Excess',
                                                y_title='Nutrient Excess',
                                                legend_title='Nutrient')
   
    st.plotly_chart(fig_left_over, use_container_width=True)
    if st.checkbox('Show nutrient excess data'):
        st.write(df_left_over)

    # fertilizer usage
    fig_nutrient_usage = ps.plot_stacked_interactive(df_fertilizer,
                                                     fig_title='Fertilizer Usage',
                                                     y_title='Fertilizer Usage',
                                                     legend_title='Fertilizer')
    st.plotly_chart(fig_nutrient_usage)
    if st.checkbox('Show fertilier amount data'):
        st.write(df_fertilizer)
        df_fertilizer_sum = df_fertilizer.sum()
        df_fertilizer_sum_df = pd.DataFrame(df_fertilizer_sum, columns=['Total'])
        st.write(df_fertilizer_sum_df)

