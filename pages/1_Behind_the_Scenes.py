# pages/1_Behind_the_Scenes.py

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Technical Details", page_icon="🛠️", layout="wide")

st.title("Behind the Scenes: The Technical Details")

# Check if a model run has been completed
if 'run_details' in st.session_state and st.session_state.run_details:
    details = st.session_state.run_details

    # Display the logs
    st.subheader("Processing Log")
    log_text = "\n".join(details.get('logs', ["No logs available."]))
    st.code(log_text, language='text')

    # Display DataFrames
    st.subheader("Data Previews")
    if 'initial_data_head' in details:
        st.write("Initial Data (first 5 rows):")
        st.dataframe(details['initial_data_head'])
    
    if 'featured_data_head' in details:
        st.write("Data After Feature Engineering (first 5 rows):")
        st.dataframe(details['featured_data_head'])

    # Display Correlation Matrix
    st.subheader("Feature Correlation Matrix")
    if 'correlation_matrix' in details:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(details['correlation_matrix'], annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    
    # Display best parameters
    st.subheader("Model Tuning Results")
    if 'best_params' in details:
        st.write("Best Hyperparameters found by GridSearchCV:")
        st.json(details['best_params'])
    
else:
    st.warning("Please run an analysis on the main page first to see the technical details.")