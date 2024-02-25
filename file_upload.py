import streamlit as st
import pandas as pd
from io import StringIO
import time
from train_m_single_m_p_pab1_test import main as train_main

st.markdown("""

# Welcome to EvoPlay_Biotrainer

**This page allows you to upload text, CSV, or TSV files and easily download generated sequences.**

**Here's how it works:**

1. **Upload a file:** Choose a file using the file uploader below.
2. **Download CSV file with generated sequences:** At the end of the programm a CSV file will be created that includes all the generated sequences
3. **Get Performance Analysis:** The generated CSV file will then be visualized here to assess performance         

""")

uploaded_file = st.file_uploader("Choose a file:", type=['csv', 'tsv', 'txt'])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[-1]
    st.write("You uploaded a", file_type, "file.")



    
   

    

