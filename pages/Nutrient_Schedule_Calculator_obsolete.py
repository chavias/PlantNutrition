import streamlit as st
import pandas as pd
import plant_scheduler.nutrients as ps
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

# File uploader for the nutrient concentration
uploaded_concentration_file = st.file_uploader("Choose file of optimal nutrient time series ...", type=["xlsx", "csv", "tsv"])
data = None  # Initialize data

if uploaded_concentration_file is not None:
    data = load_data(uploaded_concentration_file)

if data is not None and st.checkbox('Show nutrient concentration data'):
    st.subheader('Nutrient concentration')
    st.write(data)

c_fertilizer1 = np.abs(np.random.sample(11))*10
c_fertilizer2 = np.abs(np.random.sample(11))*3
c_fertilizer3 = np.abs(np.random.sample(11))*0.1

if uploaded_concentration_file is not None: 
    time_intervall_days = st.selectbox(
    "Time intervall for fertilizing?",
    list(range(1, data.shape[1]*7)),
    )

    st.write(f"You selected: {time_intervall_days} days")
    df_results = ps.calculate_fertilization_schedule_df(df=data,
                                                time_intervall_days=time_intervall_days,
                                                c_fertilizer1=c_fertilizer1,
                                                c_fertilizer2=c_fertilizer2,
                                                c_fertilizer3=c_fertilizer3)
    fig_excess = ps.plot_excess_stacked_interactive_streamlit(df_results)
    st.plotly_chart(fig_excess, use_container_width=True)