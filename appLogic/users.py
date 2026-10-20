import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import appLogic.processTax as tax#currently giving incorrect values
import appLogic.catagorizeLR as logCat
from classifier.incremental_learning_pipelines import IncrementalMultinomialNBClassifier
from classifier.pipelines import pipeline
from sklearn.naive_bayes import MultinomialNB


class User ():
    def __init__(self, username):
        def incomeToBudget(jobs_df, budget_df):
            """Add the users after tax income from jobs_df to the budget_df.
            Doesn't require working processTax module"""
            income = jobs_df[['Title', 
                                'After Tax Income', 
                                'Start Date', 
                                'End Date']]
            income['Monthly Amount'] = income['After Tax Income'].apply(lambda row: row/26*2)
            income = income.rename(columns={'Title':'Category'})
            income['Type'] = 'Income'
            budget_df = pd.concat([budget_df, income.loc[:,['Category', 'Type', 'Monthly Amount']]]).reset_index(drop=True)
            return budget_df
        
        def filter_table_to_current(df: pd.DataFrame, end_column:str):
            """filter a userdata table to only include current values"""
            df = df.loc[(df[end_column].isnull() == True) | (df[end_column] > datetime.today())]
            return df

        def format_date_columns(df: pd.DataFrame, start_column: str, end_column: str, list_of_columns:list|None = None):
            """format start and end columns as pd.datetime"""
            if list_of_columns == None:
                list_of_columns = [start_column, end_column]
            df[list_of_columns] = df[list_of_columns].apply(lambda x: pd.to_datetime(x, format="%m/%d/%Y"))
            return df

        self.username = username
        # generated variables below
        self.userData = Path.cwd() / 'userData' / self.username
        #Removed. Don't think I need to track this information
        # self.accounts = pd.read_csv(self.userData / 'Accounts.csv')
        self.budget = pd.read_csv(self.userData / 'Budget.csv')
        self.statements = self.userData / 'Statements'
        self.gl = pd.read_csv(self.userData / 'GL_2lvl.csv')
        self.gl = self.gl.loc[:, ~self.gl.columns.str.startswith('Unnamed')]
        self.gl['Location'] = self.gl['Location'].astype(str).str.strip()
        self.gl['Source'] = self.gl['Source'].astype(str).str.strip()
        self.gl['Credit'] = self.gl['Credit'].round(2)
        self.gl['Debit'] = self.gl['Debit'].round(2)
        self.gl['Balance'] = self.gl['Balance'].round(2)
        self.gl['Date'] = pd.to_datetime(self.gl['Date'])
        self.gl = self.gl.drop_duplicates(
            subset=['Date', 'Location', 'Credit', 'Debit', 'Balance', 'Source'],
            ignore_index=True,
            keep='last'
        )
        #for jobs
        # self.jobs = tax.canadaincomeTax(self.userData)
        self.jobs = pd.read_csv(self.userData / 'Jobs.csv')
        self.jobs = format_date_columns(df=self.jobs, start_column='Start Date', end_column='End Date')
        self.current_jobs = filter_table_to_current(df=self.jobs, end_column='End Date')
        #for budget
        self.budget = pd.read_csv(self.userData / 'Budget.csv')
        self.budget = incomeToBudget(jobs_df=self.jobs, budget_df=self.budget)
        self.budget = format_date_columns(df=self.budget, start_column='Start Date', end_column='End Date')
        self.current_budget = filter_table_to_current(df=self.budget, end_column='End Date') #arguabely should create self.current_budget
        self.cat = logCat.Categorizer()
    
    def read_inputs(self):
        """Read all the input files and put them in a list"""
        newData = []
        def newColumns(df):
            df['Tag1'] = None
            df['E_Transfer'] = None
            df['Source'] = str(folder.stem)
            df = df[['Date', 'Location', 'Tag1', 'Credit', 'Debit','Balance', 'Source', 'E_Transfer']]
            return df
        
        for folder in self.statements.iterdir():
            if folder.is_dir() == True:#prevent non-folders being read
                path = self.statements / folder
                for file in path.iterdir():
                    if file.suffix == '.csv':#read .csv file formats
                        df = pd.read_csv(path / file,header=None,names=['Date','Location','Credit','Debit','Balance'], parse_dates=['Date'])
                        df = newColumns(df)
                        newData.append(df)
                    elif file.suffix == '.xlsx':#read .xlsx file formats
                        df = pd.read_excel(path / file, names=['Date','Location','Credit','Debit','Balance'], parse_dates=['Date'])
                        df = newColumns(df)
                        newData.append(df)
                    else:
                        continue
            else:
                continue
        newDatadf = pd.concat(newData, ignore_index=True)
        return newDatadf

    def loadModel(self, first_time):
        """Setup the user's personal model"""
        existing_labels = list(self.gl['Tag1'].unique())
        self.clf = IncrementalMultinomialNBClassifier(
            user_labels=existing_labels,
            preprocessor=pipeline,
            nb_classifier=MultinomialNB(),
        )
        if first_time:
            existing_labels = [
                'Car', 
                'Entertainment', 
                'Finance', 
                'Food', 
                'Health', 
                'Home', 
                'Income', 
                'Personal', 
                'Spending', 
                'Transfer from', 
                'Transfer to'
            ]
            self.clf = IncrementalMultinomialNBClassifier(
                user_labels=existing_labels,
                preprocessor=pipeline,
                nb_classifier=MultinomialNB(),
            )
            self.clf.load_trained_model(path=Path.cwd() / 'classifier')

        elif not first_time:
            existing_labels = list(self.gl['Tag1'].unique())
            self.clf = IncrementalMultinomialNBClassifier(
                user_labels=existing_labels,
                preprocessor=pipeline,
                nb_classifier=MultinomialNB(),
            )
            self.clf.load_trained_model(path=self.userData)

    def predictNewData(self, newData):
        print('Predicting new categories')
        predictions = self.clf.predict_new(next_batch_X=newData)
        output_df = newData.copy()
        output_df['Tag1'] = predictions
        return output_df

    def reportPerformance(self, originalPredictions, correctedPredictions):
        comparison = correctedPredictions.merge(
            originalPredictions[['Date', 'Location', 'Credit', 'Debit', 'Balance', 'Source', 'Tag1']].rename(columns={'Tag1': 'predicted'}),
            on=['Date', 'Location', 'Credit', 'Debit', 'Balance', 'Source'],
            how='inner'
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        all_possible_labels = sorted(comparison['Tag1'].unique().tolist())
        #I should log the accuracy over time somewhere.
        # save_plot_path = self.userData / "modelPerformance"
        # today_str = datetime.today().strftime('%Y-%m-%d')

        ConfusionMatrixDisplay.from_predictions(
        comparison['Tag1'],
        comparison['predicted'],
        labels=all_possible_labels,
        xticks_rotation='vertical',
        cmap='Blues',
        ax=ax,
        colorbar=True
        )
        plt.title("User Correction Confusion Matrix")
        return fig

    def save_model(self):
        self.clf.save_model(self.userData)

    def savePredictions(self, correctedPredictions):
        #regular merge and export
        self.gl = pd.concat([self.gl, correctedPredictions],ignore_index=True)
        #data cleaning for deduplication:
        self.gl = self.gl.loc[:, ~self.gl.columns.str.startswith('Unnamed')]
        self.gl['Source'] = self.gl['Source'].astype(str).str.strip()
        self.gl['Location'] = self.gl['Location'].astype(str).str.strip()
        self.gl['Credit'] = self.gl['Credit'].round(2)
        self.gl['Debit'] = self.gl['Debit'].round(2)
        self.gl['Balance'] = self.gl['Balance'].round(2)
        self.gl['Date'] = pd.to_datetime(self.gl['Date'])
        self.gl = self.gl.fillna(0)
        self.gl = self.gl.drop_duplicates(subset=['Date','Location','Credit','Debit','Balance','Source'],ignore_index=True, keep='last')
        self.gl = self.gl.sort_values(by='Date')
        self.gl = self.gl.reset_index(drop=True)
        self.gl.to_csv(self.userData / 'GL_2lvl.csv', index=False)
    

def main():
    Mack = User('Mack')
    return Mack

if __name__ == '__main__':
    Mack = main()