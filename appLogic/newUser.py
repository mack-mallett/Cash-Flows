import streamlit as st
from pathlib import Path
import pandas as pd

class newUser ():
    def __init__(self, user_name):
        self.userName = user_name

        self.userNameList = pd.read_csv(Path.cwd() / 'userNames.csv', index_col=['UserID'])
        self.userFolder = Path.cwd() / 'userData' / f"{self.userName}"

    def _add_bucket(self):
        temp = st.session_state.temp_bucket.strip()
        if temp and temp not in st.session_state.buckets:
            st.session_state.buckets.append(temp)
        # Clear the text input for the next entry
        st.session_state.temp_bucket = ""

    def make_user(self):
        ready = False
        #User Data setup
        #Make the user's data folder
        self.userFolder.mkdir(exist_ok=True)

        #Create a fresh, 2-level General Ledger (GL_2lvl)
        self.gL = pd.DataFrame(columns = ['Date', 'Location', 'Tag1', 'Tag2', 'Credit', 'Debit', 'Balance', 'Source'])
        self.gL.to_csv(self.userFolder / 'GL_2lvl.csv')

        #Create a file for the users income streams
        self.jobs = pd.DataFrame(columns= ['Title', 'Employer', 'Yearly Income Before Deductions', 'Deductions Before Tax', 'Deductions After Tax', 'Yearly Income After Deductions', 'After Tax Income', 'Country', 'Province', 'Start Date', 'End Date'])
        self.jobs.to_csv(self.userFolder / 'Jobs.csv')

        self.inputDir = self.userFolder / 'Statements'
        self.inputDir.mkdir(exist_ok=True)

        # #Create a data sheet for user accounts
        # Removed, I don't think I need to track this information
        # self.accounts = pd.DataFrame(columns= ['Name', 'Account Number', 'Interest Rate', 'Active', 'Input Slot'])
        # self.accounts.to_csv(self.userFolder / 'Accounts.csv')

        self.budget = pd.DataFrame(columns=['Category', 'Monthly Amount', 'Type', 'Start Date', 'End Date'])
        self.budget.to_csv(self.userFolder / 'Budget.csv')

        #Save the username to a file of usernames
        self.Name = pd.DataFrame({'User Names': self.userName}, index = [0])
        self.userNameList = pd.read_csv(Path.cwd() / 'userNames.csv', index_col=['UserID'])
        self.userNameList = pd.concat([self.userNameList, self.Name])
        self.userNameList.to_csv(Path.cwd() / 'userNames.csv', index_label=['UserID'])

