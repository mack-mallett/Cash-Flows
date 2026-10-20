import streamlit as st


user = st.session_state['user']

if 'buckets' not in st.session_state:
    st.session_state['buckets'] = [folder.name for folder in user.statements.iterdir() if folder.is_dir()]
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'upload' #options: 'upload', 'classify'
if 'new_data' not in st.session_state:
    st.session_state['new_data'] = None 

def add_bucket():
    temp = st.session_state.temp_bucket.strip()
    if temp and temp not in st.session_state['buckets']:
        st.session_state['buckets'].append(temp)
    # Clear the text input for the next entry
    st.session_state.temp_bucket = ""

st.header("Add new transactions")
st.markdown(f"""
On this page you can classify new trasactions according to your previous categories.
New user? No problem. Take advantage of the pretrained model to get started, and watch 
it adjust to your preferences over time. 
""")

if st.session_state['stage'] == 'upload':

    st.markdown(f"""
    To get started, download your transaction histories from your bank's website.
    Drag and drop from your downloads folder into the buckets you defined when creating your profile.
    """)

    with st.expander("Add buckets"):
        st.text_input(
            "Please enter the accounts/buckets you wish to track. DO NOT ENTER FULL ACCOUNT NUMBERS!! (press Enter after each):",
            key="temp_bucket",
            on_change=add_bucket,
            placeholder="e.g., Savings, Checking, Investment..."
        )
        if st.session_state.buckets:
            st.markdown("### Tracked Buckets:")
            for bucket in st.session_state.buckets:
                st.markdown(f"**{bucket}**")
        else:
            st.info("No buckets added yet. Type a name above and press Enter.")
        
        if st.button("Done"):
            if st.session_state.buckets:
                st.success(f"Successfully saved {len(st.session_state.buckets)} buckets!")
                # self.userAccounts = st.session_state.buckets
                user.statements.mkdir(exist_ok=True)
                for acct in st.session_state.buckets:
                    dir = user.statements / acct
                    dir.mkdir(exist_ok=True)
            else:
                st.warning("Please add at least one bucket before clicking Done.")


    if not st.session_state['buckets']:
        st.warning("No statement folders found in your directory. Please add some account buckets first.")

    else:
        for bucket_name in st.session_state['buckets']:
            folder = user.statements / bucket_name
            folder.mkdir(exist_ok=True, parents=True)

            with st.expander(f"Bucket: `{folder.name}`", expanded=True):
                
                # Create a side-by-side layout for the uploader and clear button
                col1, col2 = st.columns([4, 1])
                
                with col2:
                    # Clear directory button unique to this bucket
                    if st.button("Clear Dir", key=f"clear_{bucket_name}", type="secondary"):
                        for file_path in folder.iterdir():
                            if file_path.is_file():
                                file_path.unlink()
                        st.success(f"Cleared all files in `{bucket_name}`!")
                        st.rerun()

                with col1:
                    # File uploader specific to this folder/bucket
                    uploaded_file = st.file_uploader(
                        f"Upload CSV for {folder.name}",
                        type="csv",
                        key=f"uploader_{folder.name}"
                    )

                # Save the uploaded file to the disk directory if provided
                if uploaded_file is not None:
                    file_path = folder / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved `{uploaded_file.name}` to `{folder.name}/`!")

                # Display files already sitting in this bucket directory
                existing_files = [f.name for f in folder.iterdir() if f.is_file()]
                if existing_files:
                    st.markdown("**Files currently in this bucket:**")
                    for fname in existing_files:
                        st.caption(f"{fname}")
                else:
                    st.caption("No files uploaded yet.")

    st.header("Finished uploading? Let's classify your purchases!")
    if st.button("Classify!"):
        if st.session_state['new_user'] == True:
            #load classifier from default params
            pass
        elif st.session_state['new_user'] == False:
            #load classifier from user params
            pass
        else:
            raise RuntimeError("Somehow user is neither new_user or not new_user. Must not have started from login screen. Reload and try from login screen")
        st.session_state['stage'] = 'classify'
        st.session_state['new_data'] = user.read_inputs()
        st.rerun()

elif st.session_state['stage'] == 'classify':
    with st.spinner("Please wait while we classify your purchases"):
        user.loadModel(
            first_time=st.session_state['new_user']
        )
        predictions= user.predictNewData(newData=st.session_state['new_data'])
    st.markdown(f"""
    Here's how your personal model thinks your purchases should be catagorized. Predictions are stored in the 'Tag1' column. 
    If this is your first time using the platform, then you are seeing predictions from our default model, which may be different
      than your personal budgeting style. To make changes, click on a cell and type your prefered catagory. This information will
        be stored in your personal model weights for future uploads! 
    """)
    corrected_predictions = st.data_editor(
        data=predictions,
        on_change='ignore',
        disabled=['Date', 'Location', 'Tag2', 'Credit', 'Debit','Balance', 'Source', 'E_Transfer']
    )

    if st.button("I'm done making changes"):
        user.savePredictions(
            correctedPredictions = corrected_predictions
            )
        conf_matrix = user.reportPerformance(
            originalPredictions = predictions,
            correctedPredictions = corrected_predictions
            )
        user.save_model()
        st.session_state['conf_matrix'] = conf_matrix

        # Show the confusion matrix and reset button only after changes are submitted
        if 'conf_matrix' in st.session_state and st.session_state['conf_matrix'] is not None:
            st.pyplot(st.session_state['conf_matrix'])
            
            # Reset button that instantly clears state and goes back to upload
            if st.button("Performance noted, reset"):
                st.session_state['stage'] = 'upload'
                st.session_state['new_data'] = None
                if 'conf_matrix' in st.session_state:
                    del st.session_state['conf_matrix']
                st.rerun()