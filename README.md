WIP cash flow analysis platform

What: Create a web service allowing users to categorize transactions from their credit card and bank statements into user defined budget categories. Users should be able to investigate their purchase history and predict future spending (to a reasonable degree)  

Who: This is for people who are interested in detailed management of personal finances, but are short on time. In the future this could grow to include couples and families with these priorities.  

Where: Using the platform is a sit-down activity, just like other budgeting tasks. The mobile friendly website is really for tablet users and users without access to their laptop.  

When: I started implementing this system for personal use as far back as 2021. This project exists in a time where Gen-Z users are moving to digital wealth management platforms like Wealthsimple, and most Banks have some version of this software (although I don't know anyone who actually uses it). There is speculation of open banking laws in the news, which would be good for a platform like this to disrupt the existing banking apps with improved user experiences.  

Why: Practice implementing ML systems into web apps. Grow as an engineer.  

TODO:
 - Compare different ML models for transaction classification. 
   - The Logistic Regression system from class, XGBoost as a strong tree method, and some deep learning method (maybe this offers another approach to encoding the POS System ID)
   - Systems being compared should be organized into 'Pipeline's
 - Build a Microservice for the best performing model (with FastAPI?)
 - Setup a website for users to use the microservice. Should be extremely simple. Should include some kind of security framework.
   - mobile friendly version
 - Build a microservice for past spending analysis with Langchain
 - Build a receipt uploader microservice
 - Develop a system to estimate the tax burden of new or changing personal income. Start with Canadian Tax Law.
