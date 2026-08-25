import os
import pandas as pd

class newUser ():
    def __init__(self):
        self.userName = str(input("Hello! Welcome to the Money Machine. Let's get you set up. Please enter your username."))
        self.userFolder = 'Accounts/userData/'+self.userName
        #User Data setup
        os.makedirs(self.userFolder)
        self.gL = pd.DataFrame(columns = ['Date', 'Location', 'Tag1', 'Tag2', 'Credit', 'Debit', 'Balance', 'Source'])
        self.gL.to_csv(self.userFolder + '/GL.csv')
        self.jobs = pd.DataFrame(columns= ['Title', 'Employer', 'Yearly Income', 'Deductions Before Tax', 'Deductions After Tax', 'Income Tax Table', 'Active'])
        self.jobs.to_csv(self.userFolder + '/Jobs.csv')
        self.accounts = pd.DataFrame(columns= ['Name', 'Account Number', 'Interest Rate', 'Active', 'Input Slot'])
        self.accounts.to_csv(self.userFolder + '/Accounts.csv')
        self.budget = pd.DataFrame(columns=['Category', 'Monthly Amount', 'Type'])
        self.budget.to_csv(self.userFolder + '/Budget.csv')

        self.Name = pd.DataFrame({'User Names': self.userName}, index = [0])
        self.userNameList = pd.read_csv('Accounts/userNames.csv')
        self.userNameList = pd.concat([self.userNameList, self.Name])
        self.userNameList.to_csv('Accounts/userNames.csv')

if __name__ == '__main__':
    newUser()

