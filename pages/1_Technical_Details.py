# pages/1_Technical_Details.py - FINAL Corrected Version

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Technical Details", page_icon="🛠️", layout="wide")

st.title("Behind the Scenes: The Technical Details")

if 'run_details' in st.session_state and st.session_state.run_details:
    details = st.session_state.run_details

    st.subheader("Processing Log")
    log_text = "\n".join(details.get('logs', ["No logs available."]))
    st.markdown(log_text)

    st.subheader("Data Previews")
    if 'initial_data_head' in details:
        st.write("**Initial Data (first 5 rows):**")
        st.dataframe(details['initial_data_head'])
    
    if 'featured_data_head' in details:
        st.write("**Data After Feature Engineering (first 5 rows):**")
        st.dataframe(details['featured_data_head'])

    st.subheader("Model Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Feature Importance**")
        if 'model' in details and 'features' in details:
            feature_importances = pd.Series(details['model'].feature_importances_, index=details['features']).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)
            ax.set_title("Model Feature Importance")
            st.pyplot(fig)

    with col2:
        st.write("**Feature Correlation Matrix**")
        if 'correlation_matrix' in details:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(details['correlation_matrix'], annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
    
    st.subheader("Model Tuning Results")
    if 'best_params' in details:
        st.write("Best Hyperparameters found by GridSearchCV:")
        st.json(details['best_params'])
    
else:
    st.warning("Please run an analysis on the main dashboard page first to see the technical details.")