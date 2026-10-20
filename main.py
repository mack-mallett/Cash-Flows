import streamlit as st
import pandas as pd
from pathlib import Path
from appLogic.users import User
from appLogic.newUser import newUser

if 'user' not in st.session_state:
    st.session_state['user'] = None
if "current_view" not in st.session_state:
  st.session_state["current_view"] = "login"  # Options: 'login', 'new_user', 'logged_in'
  if 'new_user' not in st.session_state:
    st.session_state['new_user'] = False

# new_user = st.Page("views/new_user.py", title="New User")
dashboard = st.Page("views/dashboard.py", title="Planning")
add_new = st.Page("views/add_new_purchases.py", title="Add Transactions")

def handle_login(username):
    user = username.strip() #type: ignore
    csv_path = Path.cwd() / "userNames.csv"

    if not csv_path.exists():
        st.error("User file missing.")
        st.rerun()
    usernames = pd.read_csv(csv_path, index_col=['UserID'])

    if user in usernames["User Names"].values:
        st.session_state['user'] = User(user)
        st.session_state['current_view'] = 'logged_in'
        st.rerun()
    else:
        st.error("Username not found.")

#Welcome page:
if st.session_state['current_view'] == 'login':
    if st.session_state["user"] is None:
        st.title("Welcome to Budget Bot!")
        st.subheader("Returning user? Just enter your username to get started. New User? Get started by clicking 'New User' below!")
        st.markdown(
            """
            Note on privacy: This project is not secure. You should only analyze financial data on a secure, private machine. Your data will not be uploaded to any cloud service.
        """
        )

        username = st.text_input("Username:")

        if username:
            handle_login(username)

        if st.button("New User!"):
            st.session_state['current_view'] = 'new_user'
            st.session_state['new_user'] = True
            st.rerun()
            # st.switch_page(new_user)

    else:
        st.session_state['current_view'] = 'logged_in'
        st.rerun()
        # pg = st.navigation([dashboard, add_new])
        # pg.run()

#New User
elif st.session_state['current_view'] == 'new_user':
    st.header("Welcome new user! To get started, please enter a username below:")
    st.markdown("""
    ### Notes on safety:

    This app runs entirly on your own machine. At this time, no password is required to access your account.
    Please only use this app on a secure, private machine (like the one you would feel comfortable storing a picture of your driver's licence).
    """)
    user = st.text_input("Your username here!")

    if st.button("Return to login"):
        st.session_state['current_view'] = 'login'
        st.rerun()

    if user:
        userNameRec = Path.cwd() / 'userNames.csv'
        
        if userNameRec.is_file() == False:
            new = pd.DataFrame(columns=['User Names','UserID'])
            new.to_csv(userNameRec)
        userNameList = pd.read_csv(userNameRec, index_col=['UserID'])

        if user in list(userNameList['User Names']):
            st.write('It looks like you already have an acccount! Please return to the login screen or enter a new name.')
        
        else:
            new = newUser(user)
            new.make_user()
            st.session_state.user = User(user)
            st.session_state['current_view'] = 'logged_in'
            st.rerun()


elif st.session_state['current_view'] == 'logged_in':
    pg = st.navigation([dashboard, add_new])
    pg.run()