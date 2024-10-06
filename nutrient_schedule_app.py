import streamlit as st
import pandas as pd
import plant_scheduler.nutrients as ps
import numpy as np

st.title('Nutrient schedule')

st.markdown('## Nutrient Concentration')

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
uploaded_concentration_file = st.file_uploader("Choose File ...", type=["xlsx", "csv", "tsv"])
data = None  # Initialize data

if uploaded_concentration_file is not None:
    data = load_data(uploaded_concentration_file)

if data is not None and st.checkbox('Show raw data'):
    st.subheader('Nutrient concentration')
    st.write(data)

c_fertilizer1 = np.abs(np.random.sample(11))*10
c_fertilizer2 = np.abs(np.random.sample(11))*3
c_fertilizer3 = np.abs(np.random.sample(11))*0.1
    
df_results = ps.calculate_fertilization_schedule(data_path=DATA_PATH_AVG,
                                               time_intervall_days=2,
                                               c_fertilizer1=c_fertilizer1,
                                               c_fertilizer2=c_fertilizer2,
                                               c_fertilizer3=c_fertilizer3)