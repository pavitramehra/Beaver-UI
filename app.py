import streamlit as st
import pandas as pd
import time
import numpy as np

# Set up page
st.set_page_config(page_title="Beaver: Data Cleaning using LLM's", layout="wide")
st.title("Beaver: Data Cleaning using LLM's")

# Define steps
step_titles = [
    "1. Upload Data Lake & Query Table",
    "2. Generate FD Outputs",
    "3. Calculate Entropy",
    "4. Join Prioritization",
    "5. Low Entropy Grouping",
    "6. Confidence Calculation",
    "7. QA Generation",
    "8. Train Model",
    "9. Mark Query Table Errors",
    "10. View Corrected File"
]

# Initialize session state
if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "data_lake" not in st.session_state:
    st.session_state.data_lake = {}
if "query_table" not in st.session_state:
    st.session_state.query_table = None
if "error_cells" not in st.session_state:
    st.session_state.error_cells = []

# Sidebar step navigation
st.sidebar.title("🧭 Pipeline Steps")
selected = st.sidebar.selectbox("Go to Step", list(range(len(step_titles))),
                                format_func=lambda i: step_titles[i],
                                index=st.session_state.step_index)
st.session_state.step_index = selected

# Get current step label
step = step_titles[st.session_state.step_index]
st.subheader(f"{step}")



# # Session states
# if "data_lake" not in st.session_state:
#     st.session_state.data_lake = {}

# if "query_table" not in st.session_state:
#     st.session_state.query_table = None

# STEP 1: Upload
if step == "1. Upload Data Lake & Query Table":
    # st.subheader("Upload Data Lake Files")
    lake_name = st.text_input("Name your data lake")
    files = st.file_uploader("Upload CSV files for the data lake", type="csv", accept_multiple_files=True)

    st.subheader("📄 Upload Query Table")
    query_file = st.file_uploader("Upload the query table (CSV)", type="csv")

    if files:
        for file in files:
            st.session_state.data_lake[file.name] = pd.read_csv(file)
        st.success(f"Uploaded {len(files)} data lake files for '{lake_name}'.")

    if query_file:
        st.session_state.query_table = pd.read_csv(query_file)
        st.success("Uploaded query table:")
        st.dataframe(st.session_state.query_table.head())

# STEP 2: FD Output
elif step == "2. Generate FD Outputs":
    st.subheader("🧠 Functional Dependency (FD) Generation")
    selected_lake = st.selectbox("Choose data lake", ["Architecture"])  # Hardcoded for now

    if st.button("🚀 Generate FD Outputs"):
        st.success("FD outputs generated using simulated LLM output.")
        
        fd_data = {
            "fd_architect.csv": [
                {"LHS": "id", "RHS": "name"},
                {"LHS": "name", "RHS": "gender"},
                {"LHS": "name,nationality", "RHS": "id"},
            ],
            "fd_bridge.csv": [
                {"LHS": "id", "RHS": "name"},
                {"LHS": "architect_id,name", "RHS": "id"},
            ],
            "fd_mill.csv": [
                {"LHS": "id", "RHS": "location"},
                {"LHS": "location,name", "RHS": "type"},
            ]
        }

        for file, rows in fd_data.items():
            st.markdown(f"📄 **{file}**")
            st.dataframe(pd.DataFrame(rows))

# STEP 3: Entropy
elif step == "3. Calculate Entropy":
    # st.subheader("Entropy Calculation")

    if st.button("🔍 Run Entropy Calculation"):
        st.success("Entropy scores computed for each FD file (simulated).")

        entropy_outputs = {
            "fd_architect.csv": pd.DataFrame({
                "LHS": ["id", "name", "name", "name,gender"],
                "RHS": ["name", "gender", "nationality", "gender"],
                "Entropy": [0.32, 0.05, 0.0, 0.29],
                "Score": [71.3, 8000.1, 9999.99, 64.1]
            }),

            "fd_bridge.csv": pd.DataFrame({
                "LHS": ["id", "location", "architect_id", "location,length_meters"],
                "RHS": ["name", "length_meters", "bridge_name", "length_feet"],
                "Entropy": [0.41, 0.38, 0.52, 0.27],
                "Score": [59.4, 63.2, 49.9, 75.0]
            }),

            "fd_mill.csv": pd.DataFrame({
                "LHS": ["id", "architect_id,name", "location,name", "architect_id,location"],
                "RHS": ["location", "notes", "type", "built_year"],
                "Entropy": [0.36, 0.31, 0.45, 0.28],
                "Score": [54.8, 61.5, 47.3, 68.2]
            })
        }

        for file, df in entropy_outputs.items():
            st.markdown(f"#### 📄 {file}")
            st.dataframe(df, use_container_width=True)

# STEP 4: Join Prioritization
elif step == "4. Join Prioritization":
    # st.subheader("🔗 Join Prioritization")

    if st.button("📎 Prioritize Joins"):
        join_df = pd.DataFrame({
                "Column1": ["id", "architect_id", "id"],
                "Column2": ["id", "architect_id", "architect_id"],
                "File1": ["architect.csv", "bridge.csv", "architect.csv"],
                "File2": ["mill.csv", "mill.csv", "bridge.csv"],
                "Score": [38.4375, 37.734375, 12.857142857142856]
            })
        st.success("Join prioritization complete.")
        st.dataframe(join_df)


elif step == "5. Low Entropy Grouping":
    import pandas as pd

    # st.subheader("📊 Low Entropy Groupings Table")
    st.markdown("Below are simulated low entropy LHS combinations for different target attributes:")

    low_entropy_groups = {
        "bridge_length_feet": [
            {"LHS": "bridge_name", "Entropy": 0.0, "Score": 1000},
            {"LHS": "bridge_location", "Entropy": 0.1, "Score": 850},
            {"LHS": "bridge_length_meters", "Entropy": 0.05, "Score": 950},
        ],
        "architect_name": [
            {"LHS": "architect_id", "Entropy": 0.2, "Score": 900},
            {"LHS": "architect_nationality", "Entropy": 0.15, "Score": 920},
            {"LHS": "bridge_location", "Entropy": 0.1, "Score": 940},
        ],
        "architect_gender": [
            {"LHS": "bridge_location", "Entropy": 0.12, "Score": 870},
            {"LHS": "architect_name", "Entropy": 0.18, "Score": 890},
        ],
        "architect_nationality": [
            {"LHS": "architect_id", "Entropy": 0.25, "Score": 880},
            {"LHS": "bridge_length_meters", "Entropy": 0.2, "Score": 860},
        ]
    }

    # Convert to DataFrame for display
    all_rows = []
    for target, groupings in low_entropy_groups.items():
        for entry in groupings:
            entry["Target"] = target
            all_rows.append(entry)

    df = pd.DataFrame(all_rows)[["Target", "LHS", "Entropy", "Score"]]
    st.dataframe(df, use_container_width=True)



elif step == "6. Confidence Calculation":
    # st.subheader("📈 High-Confidence Association Rules")

    confidence_data = {
        "architect_gender": pd.DataFrame({
            "architect_name": ["Zaha Hadid", "Frank Gehry"],
            "architect_gender": ["female", "male"],
            "group_count": [2, 4],
            "support": [0.1333, 0.2666],
            "count_base": [2, 4],
            "confidence": [1.0, 1.0],
            "count_target": [2, 13],
            "support_target": [0.1333, 0.8666],
            "lift": [7.5, 1.15],
            "combined_metric": [28.53, 20.46]
        }),

        "architect_name": pd.DataFrame({
            "bridge_name_bridge": [
                "Aloba Arch", "Gaotun Natural Bridge",
                "Kolob Arch", "Sipapu Natural Bridge"
            ],
            "architect_name": [
                "Mies Van Der Rohe", "Mies Van Der Rohe",
                "Zaha Hadid", "Zaha Hadid"
            ],
            "group_count": [1, 1, 1, 1],
            "support": [0.0666]*4,
            "count_base": [1]*4,
            "confidence": [1.0]*4,
            "count_target": [2]*4,
            "support_target": [0.1333]*4,
            "lift": [7.5]*4,
            "combined_metric": [14.26]*4
        })
    }

    for rhs, df in confidence_data.items():
        st.markdown(f"### 🔍 Target: `{rhs}`")
        st.dataframe(df, use_container_width=True)

elif step == "7. QA Generation":
    # st.subheader("QA Pair Generation (from Confidence Rules)")

    qa_pairs = pd.DataFrame({
        "Question": [
            "What is the architect_gender for architect_name:Zaha Hadid?",
            "What is the architect_gender for architect_name:Frank Gehry?",
            "What is the architect_name for bridge_name_bridge:Aloba Arch?",
            "What is the architect_name for bridge_name_bridge:Gaotun Natural Bridge?",
            "What is the architect_name for bridge_name_bridge:Kolob Arch?",
            "What is the architect_name for bridge_name_bridge:Sipapu Natural Bridge?"
        ],
        "Answer": [
            "female", "male",
            "Mies Van Der Rohe", "Mies Van Der Rohe",
            "Zaha Hadid", "Zaha Hadid"
        ]
    })

    st.dataframe(qa_pairs, use_container_width=True)
    st.download_button("Download QA Pairs", qa_pairs.to_csv(index=False), file_name="qa_pairs.csv")

# STEP 7: Model Training
elif step == "8. Train Model":
    # st.subheader("🧠 Train Correction Model")
    if st.button("Start Training"):
        with st.spinner("Training model... please wait"):
            import time
            time.sleep(2)  # Simulate training
        st.success("Model trained successfully! 🎉")


# STEP 8: Mark Query Table Errors
elif step == "9. Mark Query Table Errors":
    # st.subheader("🟥 Mark Errors in Query Table")

    if st.session_state.query_table is None:
        st.warning("Please upload a query table in Step 1.")
    else:
        df = st.session_state.query_table.copy()

        if "error_cells" not in st.session_state:
            st.session_state.error_cells = []

        st.markdown("### Select Cell to Mark as Error")

        selected_row = st.selectbox("Select row number", df.index)
        selected_col = st.selectbox("Select column", df.columns)

        if st.button("🔴 Mark Cell as Error"):
            cell = (selected_row, selected_col)
            if cell not in st.session_state.error_cells:
                st.session_state.error_cells.append(cell)
                st.success(f"Marked cell [{selected_row}, '{selected_col}'] as error.")

        st.markdown("### Table with Highlighted Errors")

        # Generate styled HTML table
        def highlight_errors(val, row, col):
            return f"background-color: red; color: white" if (row, col) in st.session_state.error_cells else ""

        styled_df = df.style.apply(
            lambda row: [highlight_errors(row[col], row.name, col) for col in df.columns],
            axis=1
        )

        st.dataframe(styled_df, use_container_width=True)

# STEP 9: View Corrected File (Green Highlights Only)
elif step == "10. View Corrected File":

    import pandas as pd
    import io

    # ✅ Corrected table (ground truth)
    corrected_csv = """id_x,architect_name,architect_nationality,architect_gender,architect_id,id_y,bridge_name,bridge_location,bridge_length_meters,bridge_length_feet
1,Frank Lloyd Wright,American,male,1,1,Xian Ren Qiao (Fairy Bridge),"Guangxi , China",121.0,400.0
1,Frank Lloyd Wright,American,male,1,10,Shipton's Arch,"Xinjiang , China",65.0,212.0
1,Frank Lloyd Wright,American,male,1,11,Jiangzhou Arch,"Guangxi , China",65.0,212.0
1,Frank Lloyd Wright,American,male,1,12,Hazarchishma Natural Bridge,"Bamiyan Province , Afghanistan",64.2,210.6
2,Frank Gehry,Canadian,male,2,2,Landscape Arch,"Arches National Park , Utah , USA",88.0,290.0
2,Frank Gehry,Canadian,male,2,9,Stevens Arch,"Escalante Canyon , Utah , USA",67.0,220.0
2,Frank Gehry,Canadian,male,2,13,Outlaw Arch,"Dinosaur National Monument , Colorado , USA",63.0,206.0
2,Frank Gehry,Canadian,male,2,14,Snake Bridge,"Sanostee , New Mexico , USA",62.0,204.0
3,Zaha Hadid,"Iraqi, British",female,3,3,Kolob Arch,"Zion National Park , Utah , USA",87.0,287.0
3,Zaha Hadid,"Iraqi, British",female,3,8,Sipapu Natural Bridge,"Natural Bridges National Monument , Utah , USA",69.0,225.0
4,Mies Van Der Rohe,"German, American",male,4,4,Aloba Arch,"Ennedi Plateau , Chad",76.0,250.0
4,Mies Van Der Rohe,"German, American",male,4,7,Gaotun Natural Bridge,"Guizhou , China",70.0,230.0
5,Le Corbusier,"Swiss, French",male,5,5,Morning Glory Natural Bridge,"Negro Bill Canyon , Utah , USA",74.0,243.0
5,Le Corbusier,"Swiss, French",male,5,6,Rainbow Bridge,"Glen Canyon National Recreation Area , Utah , USA",71.0,234.0
5,Le Corbusier,"Swiss, French",male,5,15,Wrather Arch,"Wrather Canyon , Arizona , USA",75.0,246.0"""

    # ❌ Original uncorrected query table
    query_csv = """id_x,architect_name,architect_nationality,architect_gender,architect_id,id_y,bridge_name,bridge_location,bridge_length_meters,bridge_length_feet
1,Frank Lloyd Wright,American,male,1,1,Xian Ren Qiao (Fairy Bridge),"Guangxi , China",121.0,400.0
1,,American,male,1,10,Shipton's Arch,"Xinjiang , China",65.0,212.0
1,Fl Wright,,male,1,11,Jzhou Arch,"Guangxi , China",65.0,212.0
1,Frank Lloyd Wright,American,male,1,12,Hahma Natural Bridge,"Bamiyan Province , Afghanistan",64.2,210.6
2,F Gehry,Canadian,female,2,2,Landscape Arch,"Arches National Park , Utah , USA",88.0,290.0
2,Frank Gehry,Canadian,male,2,9,Stes Arch,"Escalante Canyon , Utah , USA",67.0,220.0
2,Frank Ghry,Canadian,female,2,13,Outlaw Arch,"Dinosaur National Monument , Colorado , USA",63.0,206.0
2,Mies Van Der Rohe,Canadian,male,2,14,Snake Bridge,"Sanostee , New Mexico , USA",62.0,204.0
3,Zaha Hadid,"Iraqi, British",female,3,3,Kolob Arch,"Zion National Park , Utah , USA",87.0,287.0
3,Zaha Hadid,"Iraqi, British",female,3,8,Sipapu Natural Bridge,"Natural Bridges National Monument , Utah , USA",69.0,225.0
4,Ms Rohe,"German, American",male,4,4,Aloba Arch,"Ennedi Plateau , Chad",76.0,250.0
4,Mies Van Der Rohe,"German, American",male,4,7,Gaotun Natural Bridge,"Guizhou , China",70.0,230.0
5,Le Corbusier,"Swiss, French",male,5,5,Morning Glory Natural Bridge,"Negro Bill Canyon , Utah , USA",74.0,243.0
5,,"Swiss, French",male,5,6,Rainbow Bridge,"Glen Canyon National Recreation Area , Utah , USA",71.0,234.0
5,Le Corbusier,"Swiss, French",male,5,15,Wrather Arch,"Wrather Canyon , Arizona , USA",75.0,246.0"""

    corrected_df = pd.read_csv(io.StringIO(corrected_csv))
    query_df = pd.read_csv(io.StringIO(query_csv))

    # Compute mismatched cells
    mismatches = []
    for row in corrected_df.index:
        for col in corrected_df.columns:
            if query_df.at[row, col] != corrected_df.at[row, col]:
                mismatches.append((row, col))

    # Apply green style only to corrected (changed) cells
    def highlight_corrections(val, row, col):
        return "background-color: green; color: white;" if (row, col) in mismatches else ""

    styled_corrected = corrected_df.style.apply(
        lambda row: [highlight_corrections(row[col], row.name, col) for col in corrected_df.columns],
        axis=1
    )

    st.markdown("### ✅ Corrected Table with Highlighted Fixes")
    st.dataframe(styled_corrected, use_container_width=True)

st.markdown("---")
if st.session_state.step_index < len(step_titles) - 1:
    if st.button("➡️ Next Step"):
        with st.spinner("Loading next step..."):
            time.sleep(1)
        st.session_state.step_index += 1
        st.rerun()