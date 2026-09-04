# import os
from pathlib import Path
import pandas as pd

class newUser ():
    def __init__(self):
        self.userNameList = pd.read_csv(Path.cwd() / 'userNames.csv', index_col=['UserID'])
        self.userName = str(input("Hello! Welcome to the Money Machine. Let's get you set up. Please enter your username: "))
        if self.userName in list(self.userNameList['User Names']):
            return print('It looks like you already have an acccount! Please return to the login screen or enter a new name.')

        self.userFolder = Path.cwd() / 'userData' / f"{self.userName}"
        #User Data setup
        #Make the user's data folder
        self.userFolder.mkdir(exist_ok=True)
        #Create a fresh, 2-level General Ledger (GL_2lvl)
        self.gL = pd.DataFrame(columns = ['Date', 'Location', 'Tag1', 'Tag2', 'Credit', 'Debit', 'Balance', 'Source'])
        self.gL.to_csv(self.userFolder / 'GL_2lvl.csv')
        #Create a file for the users income streams
        self.jobs = pd.DataFrame(columns= ['Title', 'Employer', 'Yearly Income Before Deductions', 'Deductions Before Tax', 'Deductions After Tax', 'Yearly Income After Deductions', 'After Tax Income', 'Country', 'Province', 'Start Date', 'End Date'])
        self.jobs.to_csv(self.userFolder / 'Jobs.csv')
        #Create a directory for the user's new transaction data
        self.userInputAccounts = input("Please enter the accounts you wish to track. DO NOT ENTER FULL ACCOUNT NUMBERS!! Enter items separated by commas: \n")
        self.userAccounts = [item.strip() for item in self.userInputAccounts.split(",")]
        self.inputDir = self.userFolder / 'Statements'
        self.inputDir.mkdir(exist_ok=True)
        for acct in self.userAccounts:
            dir = self.inputDir / acct
            dir.mkdir(exist_ok=True)
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
        self.userNameList.to_csv(Path.cwd() / 'userNames.csv',index=False)

if __name__ == '__main__':
    newUser()

