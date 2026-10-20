import streamlit as st
import pandas as pd

dashboard = st.Page("views/dashboard.py", title="Planning")
add_new = st.Page("views/add_new_purchases.py", title="Add Transactions")

user = st.session_state.user

st.header("Dashboad")
st.markdown(f"""
Welcome {user.username}
In the dashboard you can see charts related to your spending.
To get started, here's a preview of the transaction data saved in your user profile.
Don't see anything here? Head over to the Add Transactions page to get started labeling your transactions.
""")

user.gl["Date"] = pd.to_datetime(user.gl["Date"])

# 1. Multiselect Slicer for Categories
categories = user.gl["Tag1"].unique()
selected_categories = st.multiselect(
    "Select Budget Categories", options=categories, default=categories
)

# 2. Calendar Date Input Slicer
min_date = user.gl["Date"].min().date()
max_date = user.gl["Date"].max().date()

selected_dates = st.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Robustly handle whatever st.date_input returns (tuple, list, or single date)
if isinstance(selected_dates, (tuple, list)):
  if len(selected_dates) == 2:
    start_date, end_date = selected_dates
  elif len(selected_dates) == 1:
    start_date = end_date = selected_dates[0]
  else:
    start_date, end_date = min_date, max_date
else:
  start_date = end_date = selected_dates

# 3. Apply the Slicers (filters) to the DataFrame
filtered_gl = user.gl[
    (user.gl["Tag1"].isin(selected_categories))
    & (user.gl["Date"] >= pd.to_datetime(start_date))
    & (user.gl["Date"] <= pd.to_datetime(end_date))
]

st.dataframe(data=filtered_gl, height=350)

st.markdown("""
    Check out your outbound spending over time by category. 
    You can zoom in and out by pinching the mousepad, like you would a touchscreen.
""")

st.subheader("Budget Categories Over Time - Outflow")
st.line_chart(
    data=filtered_gl[['Date', 'Credit', 'Tag1']], 
    x='Date',
    y='Credit',
    color='Tag1',
    x_label='Date',
    y_label='Cash Outflow'
    
    )
st.subheader("Budget Categories Over Time - Inflow")
st.line_chart(
    data=filtered_gl[['Date', 'Debit', 'Tag1']], 
    x='Date',
    y='Debit',
    color='Tag1',
    x_label='Date',
    y_label='Cash Inflow'
    
    )

st.subheader("Top Budget Categories - Outflow")
st.bar_chart(
    data = filtered_gl[['Credit', 'Tag1']],
    x='Tag1',
    y='Credit',
    x_label='Budget Category',
    y_label='Cash Outflow',
)
st.subheader("Top Budget Categories - Inflow")
st.bar_chart(
    data = filtered_gl[['Debit', 'Tag1']],
    x='Tag1',
    y='Debit',
    x_label='Budget Category',
    y_label='Cash Inflow',
)