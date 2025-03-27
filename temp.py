# import streamlit as st
# import pandas as pd
# import numpy as np

# # ⚠️ Remove or comment this line in sandbox
# # st.set_page_config(page_title="Data Processing Pipeline", layout="wide")

# st.title("🧪 Data Processing Pipeline - Simulated UI")

# # Sidebar dropdown for steps
# step = st.sidebar.selectbox("📌 Pipeline Step", [
#     "1. Upload Data Lake",
#     "2. View a File",
#     "3. Entropy Calculation",
#     "4. Join Prioritization",
#     "5. Confidence Calculation",
#     "6. QA Generation"
# ])

# # Simulate a data lake
# if "data_lake" not in st.session_state:
#     st.session_state.data_lake = {}

# # STEP 1: Upload Folder / Data Lake
# if step == "1. Upload Data Lake":
#     st.subheader("📁 Upload a Folder of CSV Files (Simulated Data Lake)")
#     files = st.file_uploader("Upload multiple CSV files", type="csv", accept_multiple_files=True)

#     if files:
#         for file in files:
#             df = pd.read_csv(file)
#             st.session_state.data_lake[file.name] = df
#             st.success(f"Loaded {file.name} with {len(df)} rows.")

#         st.info(f"📦 Total files in data lake: {len(st.session_state.data_lake)}")
#         st.write("Sample files:", list(st.session_state.data_lake.keys()))

# elif step == "2. View FD Outputs":
#     st.subheader("📊 Functional Dependencies (LLM-Based, Simulated)")

#     if not st.session_state.data_lake:
#         st.warning("Please upload CSVs in Step 1 to build your Data Lake first.")
#     else:
#         file_selected = st.selectbox("Choose a file from Data Lake", list(st.session_state.data_lake.keys()))
#         generate = st.button("🚀 Generate FD Output using LLM")

#         if generate:
#             # Simulate different FD outputs based on filename
#             if "architect" in file_selected:
#                 fd_output = [
#                     {"LHS": "id", "RHS": "name"},
#                     {"LHS": "id", "RHS": "nationality"},
#                     {"LHS": "id", "RHS": "gender"},
#                     {"LHS": "name", "RHS": "id"},
#                     {"LHS": "name", "RHS": "nationality"},
#                     {"LHS": "name", "RHS": "gender"},
#                     {"LHS": "name,nationality", "RHS": "id"},
#                     {"LHS": "name,nationality", "RHS": "gender"},
#                     {"LHS": "name,gender", "RHS": "id"},
#                     {"LHS": "name,gender", "RHS": "nationality"},
#                 ]
#             elif "bridge" in file_selected:
#                 fd_output = [
#                     {"LHS": "id", "RHS": "name"},
#                     {"LHS": "id", "RHS": "location"},
#                     {"LHS": "id", "RHS": "length_meters"},
#                     {"LHS": "id", "RHS": "length_feet"},
#                     {"LHS": "architect_id,name", "RHS": "id"},
#                     {"LHS": "architect_id,location", "RHS": "id"},
#                     {"LHS": "architect_id,length_meters", "RHS": "id"},
#                     {"LHS": "architect_id,length_feet", "RHS": "id"},
#                     {"LHS": "location,length_meters", "RHS": "id"},
#                     {"LHS": "location,length_feet", "RHS": "id"},
#                 ]
#             elif "mill" in file_selected:
#                 fd_output = [
#                     {"LHS": "id", "RHS": "architect_id"},
#                     {"LHS": "id", "RHS": "location"},
#                     {"LHS": "id", "RHS": "name"},
#                     {"LHS": "id", "RHS": "type"},
#                     {"LHS": "id", "RHS": "built_year"},
#                     {"LHS": "id", "RHS": "notes"},
#                     {"LHS": "architect_id,location", "RHS": "name"},
#                     {"LHS": "architect_id,location", "RHS": "type"},
#                     {"LHS": "architect_id,location", "RHS": "built_year"},
#                     {"LHS": "architect_id,location", "RHS": "notes"},
#                     {"LHS": "architect_id,name", "RHS": "location"},
#                     {"LHS": "architect_id,name", "RHS": "type"},
#                     {"LHS": "architect_id,name", "RHS": "built_year"},
#                     {"LHS": "architect_id,name", "RHS": "notes"},
#                     {"LHS": "location,name", "RHS": "architect_id"},
#                     {"LHS": "location,name", "RHS": "type"},
#                     {"LHS": "location,name", "RHS": "built_year"},
#                     {"LHS": "location,name", "RHS": "note"},
#                 ]
#             else:
#                 # Default simulated FDs
#                 fd_output = [
#                     {"LHS": "id", "RHS": "name"},
#                     {"LHS": "id", "RHS": "type"},
#                     {"LHS": "name", "RHS": "id"},
#                 ]

#             st.success(f"Generated FD output for `{file_selected}`")

#             # Display the result in a table
#             fd_df = pd.DataFrame(fd_output)
#             st.dataframe(fd_df)

#             # Optional: Save to session_state if needed later
#             st.session_state[f"fd_{file_selected}"] = fd_df


# # STEP 2: View Data
# elif step == "2. View a File":
#     st.subheader("🔍 Browse Data Lake")
#     if st.session_state.data_lake:
#         selected_file = st.selectbox("Choose a file to view", list(st.session_state.data_lake.keys()))
#         st.dataframe(st.session_state.data_lake[selected_file].head())
#     else:
#         st.warning("Upload CSV files first in Step 1.")

# # STEP 3: Entropy Calculation (Simulated)
# elif step == "3. Entropy Calculation":
#     st.subheader("📊 Simulated Entropy")
#     if st.session_state.data_lake:
#         selected_file = st.selectbox("Select file", list(st.session_state.data_lake.keys()))
#         df = st.session_state.data_lake[selected_file]
#         target_col = st.selectbox("Select Target Column (RHS)", df.columns)
#         st.markdown("📉 Entropy Output (simulated)")
#         entropy_df = pd.DataFrame({
#             "group_combination": [["age"], ["city"], ["name", "city"]],
#             "average_entropy": [0.32, 0.28, 0.15],
#             "score": [12.1, 14.6, 20.3]
#         })
#         st.dataframe(entropy_df)
#     else:
#         st.warning("Upload files first in Step 1.")

# # STEP 4: Join Prioritization (Simulated)
# elif step == "4. Join Prioritization":
#     st.subheader("🔗 Simulated Join Prioritization")
#     st.markdown("Showing top joins based on entropy + Jaccard (mocked data):")
#     join_df = pd.DataFrame({
#         "Column1": ["id", "name", "city"],
#         "Column2": ["user_id", "fullname", "city_name"],
#         "File1": ["users.csv", "users.csv", "users.csv"],
#         "File2": ["orders.csv", "profiles.csv", "locations.csv"],
#         "Score": [9.5, 7.8, 6.1]
#     })
#     st.dataframe(join_df)

# # STEP 5: Confidence Calculation (Simulated)
# elif step == "5. Confidence Calculation":
#     st.subheader("📈 Simulated Confidence Rules")
#     conf_df = pd.DataFrame({
#         "LHS": [["age", "city"], ["name"]],
#         "RHS": ["score", "city"],
#         "support": [0.55, 0.75],
#         "confidence": [0.89, 0.92],
#         "lift": [1.3, 1.7]
#     })
#     st.dataframe(conf_df)

# # STEP 6: QA Generation (Simulated)
# elif step == "6. QA Generation":
#     st.subheader("🤖 QA Pair Generation (Simulated)")
#     qa_df = pd.DataFrame({
#         "Question": [
#             "What is the score for name:Alice and city:New York?",
#             "What is the city for name:Bob?"
#         ],
#         "Answer": ["89", "Los Angeles"]
#     })
#     st.dataframe(qa_df)
#     st.download_button("⬇️ Download QA Pairs", qa_df.to_csv(index=False), "qa_pairs.csv")

import streamlit as st
import pandas as pd

st.title("🧪 End-to-End Data Pipeline Simulator")

# Step selection in main body
step = st.selectbox("📌 Select Step", [
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
])

# Session states
if "data_lake" not in st.session_state:
    st.session_state.data_lake = {}

if "query_table" not in st.session_state:
    st.session_state.query_table = None

# STEP 1: Upload
if step == "1. Upload Data Lake & Query Table":
    st.subheader("📁 Upload Data Lake Files")
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
    st.subheader("📉 Entropy Calculation")

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
    st.subheader("🔗 Join Prioritization")

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
    st.subheader("🧠 Low Entropy Grouping Network")
    st.markdown("Select a target attribute to see its low entropy LHS contributors.")

    import matplotlib.pyplot as plt
    import networkx as nx

    # Hardcoded low entropy groupings
    low_entropy_data = {
        "bridge_length_feet": [
            ("bridge_name",),
            ("bridge_location",),
            ("bridge_length_meters",)
        ],
        "architect_name": [
            ("architect_id",),
            ("architect_nationality",),
            ("bridge_location",)
        ],
        "architect_gender": [
            ("bridge_location",),
            ("architect_name",)
        ],
        "architect_nationality": [
            ("architect_id",),
            ("bridge_length_meters",)
        ],
        "bridge_location": [
            ("bridge_length_feet",),
            ("bridge_name",)
        ]
    }

    # Dropdown for selecting the target column
    selected_target = st.selectbox("🎯 Select Target Attribute", list(low_entropy_data.keys()))

    # Build the graph
    G = nx.DiGraph()
    lhs_list = low_entropy_data[selected_target]
    for lhs in lhs_list:
        for col in lhs:
            G.add_edge(col, selected_target)

    # Plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["#A2C8EC" if node != selected_target else "#8DE48A" for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2800)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#555555", width=2)

    plt.title(f"Low Entropy LHS → {selected_target}", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


elif step == "6. Confidence Calculation":
    st.subheader("📈 High-Confidence Association Rules")

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
    st.subheader("🤖 QA Pair Generation (from Confidence Rules)")

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
    st.download_button("📥 Download QA Pairs", qa_pairs.to_csv(index=False), file_name="qa_pairs.csv")

# STEP 7: Model Training
elif step == "8. Train Model":
    st.subheader("🧠 Train Correction Model")
    if st.button("🤖 Start Training"):
        with st.spinner("Training model... please wait"):
            import time
            time.sleep(2)  # Simulate training
        st.success("Model trained successfully! 🎉")


# STEP 8: Mark Query Table Errors
elif step == "9. Mark Query Table Errors":
    st.subheader("🟥 Mark Errors in Query Table")

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
    st.subheader("✅ Corrected File with Highlights (Green = Fixed)")

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

