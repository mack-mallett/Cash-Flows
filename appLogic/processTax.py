import pandas as pd
from pathlib import Path

#currently giving incorrect values!!
def canadaincomeTax(userFolder: Path):
    """Determine the after tax income for the user by juristiction. adds column: 'After Tax Income'"""
    jobs = pd.read_csv(userFolder / 'Jobs.csv')
    federal = jobs.groupby('Country', as_index=False).sum()
    provincial = jobs.groupby('Province', as_index=False).sum()
    juristictionIncome = {'Canada': 0, 
                        'BC': 0}
    juristictionTables = {'Canada': Path.cwd() / 'taxTables' / 'canadaIncomeTax2024.csv', 
                        'BC': Path.cwd() / 'taxTables' / 'bcIncomeTax2024.csv'}

    def incomeTax(groupby, gbycolumn):
        """function for determining juristictional income"""
        for p in range(len(groupby.index)):
            taxTable = pd.read_csv(juristictionTables[groupby[gbycolumn][p]])
            bftIncome = int(groupby['Yearly Income Before Deductions'])
            tax = 0
            s = 1
            for r in range(len(taxTable)):
                if s < len(taxTable) - 1:
                    s = r+1
                else:
                    s = r
                incomeThreshold = int(taxTable['Lower Taxable Income Threshold'].iloc[s])
                if bftIncome <= 0:
                    pass
                elif bftIncome <= float(taxTable['Lower Taxable Income Threshold'].iloc[r]):
                    tax = tax + bftIncome  * float(taxTable['Tax Rate'].iloc[r])
                    bftIncome = bftIncome - incomeThreshold
                else:
                    tax = tax + taxTable['Lower Taxable Income Threshold'].iloc[s] * float(taxTable['Tax Rate'].iloc[r])
                    bftIncome = bftIncome - incomeThreshold
        return float(tax)

    jobs['Country Tax'] = (jobs['Yearly Income After Deductions']/sum(jobs['Yearly Income After Deductions'])) * incomeTax(federal, 'Country')
    for p in provincial['Province'].unique():
        a = provincial.loc[provincial['Province'] == p]
        jobs['Province Tax'] = (jobs['Yearly Income After Deductions']/sum(jobs['Yearly Income After Deductions'])) * incomeTax(provincial, 'Province')
    jobs['Total Tax'] = jobs['Country Tax'] + jobs['Province Tax']
    jobs['After Tax Income'] = jobs['Yearly Income After Deductions'] - jobs['Total Tax']
    
    return jobs