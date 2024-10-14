import streamlit as st

st.set_page_config(
    page_title="Plant Nutrition",
    page_icon="🌿",
)

st.write("# Welcome to the Plant Nutrition Application! 👋")

st.sidebar.success("Select a analysis above.")

st.markdown(
    """
   The Plant Nutrition Application allows for simple analysis of plant nutrition data.

   ### References
    - [DRIS Index](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/diagnosis-and-recommendation-integrated-system)
"""
)