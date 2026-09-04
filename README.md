# WIP cash flow analysis platform

### About

What: Create a web service allowing users to categorize transactions from their credit card and bank statements into user defined budget categories. Users should be able to investigate their purchase history and predict future spending (to a reasonable degree)  

Who: This is for people who are interested in detailed management of personal finances, but are short on time. In the future this could grow to include couples and families with these priorities.  

Where: Using the platform is a sit-down activity, just like other budgeting tasks. The mobile friendly website is really for tablet users and users without access to their laptop.  

When: I started implementing this system for personal use as far back as 2021. This project exists in a time where Gen-Z users are moving to digital wealth management platforms like Wealthsimple, and most Banks have some version of this software (although I don't know anyone who actually uses it). There is speculation of open banking laws in the news, which would be good for a platform like this to disrupt the existing banking apps with improved user experiences.  

Why: Practice implementing ML systems into web apps. Grow as an engineer.  

## How to Run:
The best way to run this project is through a virtual environment on your machine. Setup your venv with your favorite environment manager, and import the necessary libraries through the requirements.txt file.

### Before starting!!
Note that this project does not include any security measure at this time. You should only process data on a secure, private machine.

### Starting
 - The main entry point for this project is called startup_menu.py. Run this file to begin. If this is your first time running the project, create a new file structure using the 'new user' prompts. You can then login and begin to take advantage of the transaction catagorization system. 
 - Export the files you wish to catagorize from your financial instinution in .csv format. This is your own personal information, and you should be careful with it! This project does not connect to the internet, and instead stores updated model weights directly on your computer in the project directory.
 - Add each file to its respective directory under userData / [your user name] / Statements
 - Select 'admin' from the task menu, and then 'add purchases'. You will be walked through catagorizing your own transaction data! The first time you do this you can take advantage of the pre-trained model weights available in this project.

### TODO:
 - Compare different ML models for transaction classification. 
   - The Logistic Regression system from class, XGBoost as a strong tree method, and some deep learning method (maybe this offers another approach to encoding the POS System ID)
   - Systems being compared should be organized into 'Pipeline's
 - Upgrade startup_menu to a simple streamlit interface.
 - Transition project to Ubuntu Server, begin containerization process.
 - Build a Microservice for the best performing model (with FastAPI?)
 - Setup a website for users to use the microservice. Should be extremely simple. Should include some kind of security framework.
   - mobile friendly version
 - Build a microservice for past spending analysis with Langchain
 - Build a receipt uploader microservice
 - Develop a system to estimate the tax burden of new or changing personal income. Start with Canadian Tax Law.
