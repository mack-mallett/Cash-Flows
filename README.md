# WIP Budget Bot

### About

What: Create a web service allowing users to categorize transactions from their credit card and bank statements into user defined budget categories. Users should be able to investigate their purchase history and predict future spending (to a reasonable degree)  

Who: This is for people who are interested in detailed management of personal finances, but are short on time. In the future this could grow to include couples and families with these priorities. Current primary user is myself.

Where: Using the platform is a sit-down activity, just like other budgeting tasks. The mobile friendly website is really for tablet users and users without access to their laptop.  

When: I started implementing this system for personal use as far back as 2021. This project exists in a time where Gen-Z users are moving to digital wealth management platforms like Wealthsimple, and most Banks have some version of this software (although I don't know anyone who actually uses it). There is speculation of open banking laws in the news, which would be good for a platform like this to disrupt the existing banking apps with improved user experiences.  

Why: Practice implementing ML systems into web apps. Grow as an engineer.  

## How to Run:
The best way to run this project is through a virtual environment on your machine. Setup your venv with your favorite environment manager, and import the necessary libraries through the requirements.txt file using the command `pip install -r requirements.txt`

### Before starting!!
Note that this project does not include any security measure at this time. You should only process data on a secure, private machine.

### Starting
 - The main entry point for this project is called main.py. To run this file, enter `streamlit run main.py` in the terminal. A locally hosted website will appear in your browser. This site is not connected to the internet, it only serves as an interface
    - If this is your first time running the project, use the "New User" button below the username entry bar.
 - Head to the "Add Transactions" window and follow prompts to start classifying your transactions.
 - Export the files you wish to catagorize from your financial instinution in .csv format. This is your own personal information, and you should be careful with it! This project does not connect to the internet, and instead stores updated model weights directly on your computer in the project directory. I have tried exporting from TD and CIBC online banking portals, and both produce the required columns.
 - After you have finished following the onscreen prompts you can head over to the "Planning" page to see preliminary insights. 
    - If you have a prefered visulization platform you can still use this service. Find your classified transactions under userData / [yourname] / GL_2lvl.csv

