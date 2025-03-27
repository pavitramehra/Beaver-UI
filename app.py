
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import io

# Set up page
st.set_page_config(page_title="Beaver: Data Cleaning using LLM", layout="wide")
st.title("🧹 Beaver: End-to-End Data Cleaning using LLMs")

# Shared states
if "data_lake" not in st.session_state:
    st.session_state.data_lake = {}
if "query_table" not in st.session_state:
    st.session_state.query_table = None
if "error_cells" not in st.session_state:
    st.session_state.error_cells = []
if "offline_step" not in st.session_state:
    st.session_state.offline_step = 0
if "online_step" not in st.session_state:
    st.session_state.online_step = 0

# Tabs for Offline and Online phase
tab1, tab2 = st.tabs(["🧠 Offline Phase", "🖥️ Online Phase"])

# ----------------------------- OFFLINE PHASE -----------------------------
with tab1:
    st.markdown("### 🔧 Offline Phase: Upload your Data Lake or input tables on which the model will be trained.")
    steps = [
        "Upload Data Lake",
        "Generate FD Outputs",
        "FD Entropy Calculation",
        "Join Prioritization",
        "Low Entropy Grouping",
        "High Confidence Grouping Values",
        "QA Generation",
        "Train Model"
    ]
    step = st.selectbox("Select Offline Step", steps, index=st.session_state.offline_step)

    
    if step == "Upload Data Lake":
        st.info(
    """
    💾 **What's happening in this step?**

    Upload one or more CSV files to create your **data lake**.  
    These files will be used for downstream tasks like FD generation, entropy calculation, and QA training.
    """
)
        lake_name = st.text_input("Name your data lake")
        files = st.file_uploader("Upload CSV files for the data lake", type="csv", accept_multiple_files=True)
        if files:
            for file in files:
                st.session_state.data_lake[file.name] = pd.read_csv(file)
            st.success(f"Uploaded {len(files)} files to data lake '{lake_name}'.")

    elif step == "Generate FD Outputs":
        st.subheader("🧠 Functional Dependency (FD) Generation")
        selected_lake = st.selectbox("Choose data lake", ["Architecture"])  # Hardcoded for now

            # Info section to guide the user
        st.info(
            """
            🔍 **How does this work?**

            In this step, we generate **functional dependency (FD)** candidates by prompting a large language model (LLM) on each table in the selected data lake. 
            The model analyzes table headers and sample records to suggest meaningful LHS → RHS relationships that are likely to hold in the data.

            This forms the foundation for later steps like entropy analysis, join prioritization, and automated data cleaning.
            """
        )
        if st.button("🚀 Generate FD Outputs"):
            st.success("FD outputs generated using simulated LLM output.")
            
            fd_data = {
                "fd_architect.csv": [
                    {"LHS": "id", "RHS": "name"},
                    {"LHS": "id", "RHS": "nationality"},
                    {"LHS": "name,nationality", "RHS": "gender"},
                ],
                "fd_bridge.csv": [
                    {"LHS": "id", "RHS": "name"},
                    {"LHS": "id", "RHS": "location"},
                ],
                "fd_mill.csv": [
                    {"LHS": "id", "RHS": "location"},
                    {"LHS": "location,name", "RHS": "type"},
                ]
            }

            for file, rows in fd_data.items():
                st.markdown(f"📄 **{file}**")
                st.dataframe(pd.DataFrame(rows))

    elif step == "FD Entropy Calculation":
       

        st.subheader("📊 Entropy Values for FD groups for each Table in Data Lake")
        # st.markdown("Below are simulated low entropy LHS combinations for different target attributes:")

        # Explanation for this step
        st.info(
            """
            📘 **What's happening in this step?**

            After generating Functional Dependency (FD) candidates, we now compute the **entropy** of each LHS → RHS pair.

            - **Entropy** quantifies uncertainty: lower entropy means that the LHS strongly determines the RHS.
            - We use entropy as a proxy for how "clean" or reliable a dependency is.
            - A lower entropy + higher score combination is considered more valuable for downstream tasks like joins and cleaning.

            Below are simulated entropy values for different target (RHS) attributes based on their corresponding LHS candidates.
            """
        )
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

        # Display each group as a separate table
        for target_attr, group_data in low_entropy_groups.items():
            st.markdown(f"### 🎯 Target Attribute: `{target_attr}`")
            df = pd.DataFrame(group_data)
            st.table(df)


        # # Convert to DataFrame for display
        # all_rows = []
        # for target, groupings in low_entropy_groups.items():
        #     for entry in groupings:
        #         entry["Target"] = target
        #         all_rows.append(entry)

        # df = pd.DataFrame(all_rows)[["LHS", "Target", "Entropy", "Score"]]
        # st.dataframe(df, use_container_width=True)


    elif step == "Join Prioritization":

        st.info(
        """
        📘 **What’s happening here?**

        We prioritize possible joins between tables based on overlapping column names and FD scores. 
        Higher scores indicate stronger, cleaner join paths that are preferred during integration.
        """
    )
        st.markdown("### Simulated join priority table")
        join_df = pd.DataFrame({
                "Column1": ["id", "architect_id", "id"],
                "Column2": ["id", "architect_id", "architect_id"],
                "File1": ["architect.csv", "bridge.csv", "architect.csv"],
                "File2": ["mill.csv", "mill.csv", "bridge.csv"],
                "Score": [38.4375, 37.734375, 12.857142857142856]
            })
        st.success("Join prioritization complete.")
        st.dataframe(join_df)

    elif step == "Low Entropy Grouping":

        st.info(
        """
        📘 **What's happening in this step?**

        We consider all possible groupings for each target attribute based on **low entropy combinations**.

        These groupings are strong predictors after merging datasets, and help in accurately inferring the target attribute.
        """
    )
       
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

        df = pd.DataFrame(all_rows)[["LHS", "Target", "Entropy", "Score"]]
        st.dataframe(df, use_container_width=True)


    elif step == "High Confidence Grouping Values":
        st.info(
        """
        📘 **What's happening here?**

        We extract high-confidence rules where a specific LHS value reliably predicts a target attribute (RHS). 
        These rules are scored using metrics like support, confidence, lift, and a combined score.

        ✅ This helps in **filtering out noisy or ambiguous data** and **preserving only strong, high-quality QA pairs**, 
        which are crucial for training robust models or generating clean answers.
        """
    )
        
        st.markdown("### High-confidence rules and generated QA pairs")
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
        
    elif step == "QA Generation":
        # st.subheader("QA Pair Generation (from Confidence Rules)")

        st.info(
    """
    🧠 **What's happening in this step?**

    We generate **natural language QA pairs** from high-confidence rules.

    These pairs serve as training or evaluation data for LLMs, helping the model learn how to predict target values based on reliable attribute groupings.
    """
)


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

    elif step == "Train Model":

        st.info(
    """
    🤖 **What's happening in this step?**

    We fine-tune a **T5-based language model** using the generated QA pairs.

    The model learns to map natural language questions to accurate answers, enabling it to predict missing or incorrect values during the cleaning phase.
    """
)
        if st.button("🤖 Start Training"):
            with st.spinner("Training the model..."):
                time.sleep(2)
            st.success("Model trained and ready for cleaning!")

    if st.session_state.offline_step < len(steps) - 1:
        if st.button("➡️ Next Offline Step"):
            st.session_state.offline_step += 1
            st.rerun()

# ----------------------------- ONLINE PHASE -----------------------------
with tab2:
    st.markdown("### ✨ Online Phase: Upload dirty table and clean using a pre-trained model.")
    online_steps = ["Upload Query Table & Select Model", "Mark Errors", "View Cleaned Output"]
    step = st.selectbox("Select Online Step", online_steps, index=st.session_state.online_step)

    if step == "Upload Query Table & Select Model":
        st.info("Upload the dirty table you want to clean and select a model baseline.")
        query = st.file_uploader("Upload Dirty Query Table", type="csv", key="query_table_upload")
        model = st.selectbox("Choose Cleaning Baseline", ["Beaver","Jellyfish", "RetClean", "Cocoon"])

        if query:
            st.session_state.query_table = pd.read_csv(query)
            st.dataframe(st.session_state.query_table)

    elif step == "Mark Errors":
        if st.session_state.query_table is None:
            st.warning("Please upload a query table first.")
        else:
            df = st.session_state.query_table.copy()
            selected_row = st.selectbox("Select row", df.index)
            selected_col = st.selectbox("Select column", df.columns)
            if st.button("Mark Cell as Error"):
                cell = (selected_row, selected_col)
                if cell not in st.session_state.error_cells:
                    st.session_state.error_cells.append(cell)
            st.markdown("### Table with Marked Errors")
            def highlight(val, row, col):
                return "background-color: red; color: white" if (row, col) in st.session_state.error_cells else ""
            styled_df = df.style.apply(lambda row: [highlight(row[col], row.name, col) for col in df.columns], axis=1)
            st.dataframe(styled_df)

    # elif step == "View Cleaned Output":
        
    #     # if st.session_state.query_table is not None:
    #     st.subheader("🧼 Clean Your Query Table")
    #     import io

    #      # ✅ Corrected table (ground truth)
    #     corrected_csv = """id_x,architect_name,architect_nationality,architect_gender,architect_id,id_y,bridge_name,bridge_location,bridge_length_meters,bridge_length_feet
    # 1,Frank Lloyd Wright,American,male,1,1,Xian Ren Qiao (Fairy Bridge),"Guangxi , China",121.0,400.0
    # 1,Frank Lloyd Wright,American,male,1,10,Shipton's Arch,"Xinjiang , China",65.0,212.0
    # 1,Frank Lloyd Wright,American,male,1,11,Jiangzhou Arch,"Guangxi , China",65.0,212.0
    # 1,Frank Lloyd Wright,American,male,1,12,Hazarchishma Natural Bridge,"Bamiyan Province , Afghanistan",64.2,210.6
    # 2,Frank Gehry,Canadian,male,2,2,Landscape Arch,"Arches National Park , Utah , USA",88.0,290.0
    # 2,Frank Gehry,Canadian,male,2,9,Stevens Arch,"Escalante Canyon , Utah , USA",67.0,220.0
    # 2,Frank Gehry,Canadian,male,2,13,Outlaw Arch,"Dinosaur National Monument , Colorado , USA",63.0,206.0
    # 2,Frank Gehry,Canadian,male,2,14,Snake Bridge,"Sanostee , New Mexico , USA",62.0,204.0
    # 3,Zaha Hadid,"Iraqi, British",female,3,3,Kolob Arch,"Zion National Park , Utah , USA",87.0,287.0
    # 3,Zaha Hadid,"Iraqi, British",female,3,8,Sipapu Natural Bridge,"Natural Bridges National Monument , Utah , USA",69.0,225.0
    # 4,Mies Van Der Rohe,"German, American",male,4,4,Aloba Arch,"Ennedi Plateau , Chad",76.0,250.0
    # 4,Mies Van Der Rohe,"German, American",male,4,7,Gaotun Natural Bridge,"Guizhou , China",70.0,230.0
    # 5,Le Corbusier,"Swiss, French",male,5,5,Morning Glory Natural Bridge,"Negro Bill Canyon , Utah , USA",74.0,243.0
    # 5,Le Corbusier,"Swiss, French",male,5,6,Rainbow Bridge,"Glen Canyon National Recreation Area , Utah , USA",71.0,234.0
    # 5,Le Corbusier,"Swiss, French",male,5,15,Wrather Arch,"Wrather Canyon , Arizona , USA",75.0,246.0"""

    #     # ❌ Original uncorrected query table
    #     query_csv = """id_x,architect_name,architect_nationality,architect_gender,architect_id,id_y,bridge_name,bridge_location,bridge_length_meters,bridge_length_feet
    # 1,Frank Lloyd Wright,American,male,1,1,Xian Ren Qiao (Fairy Bridge),"Guangxi , China",121.0,400.0
    # 1,,American,male,1,10,Shipton's Arch,"Xinjiang , China",65.0,212.0
    # 1,Fl Wright,,male,1,11,Jzhou Arch,"Guangxi , China",65.0,212.0
    # 1,Frank Lloyd Wright,American,male,1,12,Hahma Natural Bridge,"Bamiyan Province , Afghanistan",64.2,210.6
    # 2,F Gehry,Canadian,female,2,2,Landscape Arch,"Arches National Park , Utah , USA",88.0,290.0
    # 2,Frank Gehry,Canadian,male,2,9,Stes Arch,"Escalante Canyon , Utah , USA",67.0,220.0
    # 2,Frank Ghry,Canadian,female,2,13,Outlaw Arch,"Dinosaur National Monument , Colorado , USA",63.0,206.0
    # 2,Mies Van Der Rohe,Canadian,male,2,14,Snake Bridge,"Sanostee , New Mexico , USA",62.0,204.0
    # 3,Zaha Hadid,"Iraqi, British",female,3,3,Kolob Arch,"Zion National Park , Utah , USA",87.0,287.0
    # 3,Zaha Hadid,"Iraqi, British",female,3,8,Sipapu Natural Bridge,"Natural Bridges National Monument , Utah , USA",69.0,225.0
    # 4,Ms Rohe,"German, American",male,4,4,Aloba Arch,"Ennedi Plateau , Chad",76.0,250.0
    # 4,Mies Van Der Rohe,"German, American",male,4,7,Gaotun Natural Bridge,"Guizhou , China",70.0,230.0
    # 5,Le Corbusier,"Swiss, French",male,5,5,Morning Glory Natural Bridge,"Negro Bill Canyon , Utah , USA",74.0,243.0
    # 5,,"Swiss, French",male,5,6,Rainbow Bridge,"Glen Canyon National Recreation Area , Utah , USA",71.0,234.0
    # 5,Le Corbusier,"Swiss, French",male,5,15,Wrather Arch,"Wrather Canyon , Arizona , USA",75.0,246.0"""

    #     corrected_df = pd.read_csv(io.StringIO(corrected_csv))
    #     query_df = pd.read_csv(io.StringIO(query_csv))

    #     # Simulated file dropdown
    #     query_file_options = ["query_dirty.csv"]
    #     selected_query = st.selectbox("Select Query File", query_file_options)

    #     if st.button("🚀 Clean File"):
    #         with st.spinner("Cleaning file... please wait..."):
    #             time.sleep(2)

    #         st.success(f"✅ File `{selected_query}` cleaned successfully!")

    #         # Detect mismatches
    #         mismatches = [
    #             (row, col)
    #             for row in corrected_df.index
    #             for col in corrected_df.columns
    #             if query_df.at[row, col] != corrected_df.at[row, col]
    #         ]

    #         # Highlight only corrected cells
    #         def highlight_corrections(val, row, col):
    #             return "background-color: green; color: white;" if (row, col) in mismatches else ""

    #         styled_corrected = corrected_df.style.apply(
    #             lambda row: [highlight_corrections(row[col], row.name, col) for col in corrected_df.columns],
    #             axis=1
    #         )

    #         st.markdown("### ✅ Corrected Table with Highlighted Fixes")
    #         st.dataframe(styled_corrected, use_container_width=True)

    elif step == "View Cleaned Output":
        st.subheader("🧼 Clean Your Query Table")

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


        st.info(
            """
            📘 **What's happening in this step?**

            We use the fine-tuned model to clean your query table by predicting missing or incorrect values.  
            You can also compare different cleaning baselines using F1, Precision, and Recall (if ground truth is available).
            """
        )

        # Model selection dropdown
        selected_model = st.selectbox("Select Cleaning Model", ["Beaver", "Cocoon", "RetClean", "Jellyfish"])

        # Query file selector (simulated)
        query_file_options = ["query_dirty.csv"]
        selected_query = st.selectbox("Select Query File", query_file_options)

        if st.button("🚀 Clean File"):
            with st.spinner("Cleaning file... please wait..."):
                time.sleep(2)

            st.success(f"✅ File `{selected_query}` cleaned using `{selected_model}`!")

            # Compare with ground truth
            corrected_df = pd.read_csv(io.StringIO(corrected_csv))
            query_df = pd.read_csv(io.StringIO(query_csv))

            mismatches = [
                (row, col)
                for row in corrected_df.index
                for col in corrected_df.columns
                if query_df.at[row, col] != corrected_df.at[row, col]
            ]

            def highlight_corrections(val, row, col):
                return "background-color: green; color: white;" if (row, col) in mismatches else ""

            styled_corrected = corrected_df.style.apply(
                lambda row: [highlight_corrections(row[col], row.name, col) for col in corrected_df.columns],
                axis=1
            )

            st.markdown("### ✅ Corrected Table with Highlighted Fixes")
            st.dataframe(styled_corrected, use_container_width=True)

        if st.button("📊 Compare Baselines (F1/Precision/Recall)"):
            import matplotlib.pyplot as plt

            metrics = ["F1 Score", "Precision", "Recall"]
            models = ["Beaver", "Coocon", "RetClean"]
            scores = {
                "F1 Score": [0.91, 0.85, 0.78],
                "Precision": [0.93, 0.87, 0.80],
                "Recall": [0.90, 0.84, 0.76]
            }
            import plotly.graph_objects as go

            for metric in metrics:
                fig = go.Figure(
                    data=[go.Bar(x=models, y=scores[metric], text=scores[metric], textposition='auto')],
                )
                fig.update_layout(
                    title=f"{metric} Comparison",
                    width=50,  # control width
                    height=300,  # control height
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig)

             
        # # Select file from previously uploaded files
        # uploaded_files = [f for f in st.session_state.get("uploaded_query_files", [])]
       
        # selected_query_file = st.selectbox("Select Query File", uploaded_files)

        # if st.button("🚀 Clean File"):
        #     with st.spinner("Cleaning file..."):
        #         time.sleep(2)

        #     # Simulate loading cleaned data
        #     st.success(f"Cleaned version of `{selected_query_file}`:")
        #     cleaned_df = pd.read_csv(st.session_state[selected_query_file])
        #     st.dataframe(cleaned_df)
       
        # # Compute mismatched cells
        # mismatches = []
        # for row in corrected_df.index:
        #     for col in corrected_df.columns:
        #         if query_df.at[row, col] != corrected_df.at[row, col]:
        #             mismatches.append((row, col))

        # # Apply green style only to corrected (changed) cells
        # def highlight_corrections(val, row, col):
        #     return "background-color: green; color: white;" if (row, col) in mismatches else ""

        # styled_corrected = corrected_df.style.apply(
        #     lambda row: [highlight_corrections(row[col], row.name, col) for col in corrected_df.columns],
        #     axis=1
        # )

        # st.markdown("### ✅ Corrected Table with Highlighted Fixes")
        # st.dataframe(styled_corrected, use_container_width=True)


    if st.session_state.online_step < len(online_steps) - 1:
        if st.button("➡️ Next Online Step"):
            st.session_state.online_step += 1
            st.rerun()
