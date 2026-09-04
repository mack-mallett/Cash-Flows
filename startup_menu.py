from appLogic.users import User
from appLogic.newUser import newUser
import pandas as pd
from pathlib import Path

def menu_else():
    print()
    print('Enter a valid option!')

def get_option(available_options, option_type):
    while True:
        print()
        print(f'Please choose a(n) {option_type}')
        for option in available_options:
            print('- ' + option)
        
        print()
        user_option = input('Option: ').strip().lower()
        if user_option in available_options:
            return user_option
        
        menu_else()


def add_purchases_menu(user):
    while True:
        print(
            "\nHave you confirmed that all spreadsheets are filled in properly, "
            "and are in the right place?"
        )
        user_input = get_option(("yes", "no"), "yes or no")
        if user_input == "yes":
            user.updateAccounts()
            print("Your General Ledger has been updated with the latest information.")
            break
        elif user_input == "no":
            print("\nPlease make adjustments and come back.")
            break


def admin_tasks_menu(user):
    admin_options = (
        "add purchases",
        "update budget",
        "update your jobs",
        "exit to role selection",
    )
    while True:
        admin_option = get_option(admin_options, "pocket book function")

        if admin_option == "add purchases":
            add_purchases_menu(user)
        elif admin_option == "update budget":
            print("\nCreate a module for updating the budget")
        elif admin_option == "update your jobs":
            print("\nCreate a module for updating the jobs")
        elif admin_option == "exit to role selection":
            break


def role_selection_menu(user):
    role_options = ("admin", "analysis", "logout")
    while True:
        role_option = get_option(role_options, "group of tasks")

        if role_option == "admin":
            admin_tasks_menu(user)
        elif role_option == "analysis":
            print("\nCreate an Analysis Module. Start with cash flows!")
        elif role_option == "logout":
            break


def handle_login():
    username = input("\nEnter Username: ").strip()
    csv_path = Path.cwd() / "userNames.csv"

    if not csv_path.exists():
        print("\nUser file missing.")
        return

    usernames = pd.read_csv(csv_path)
    if username in usernames["User Names"].values:
        print(f"\nWelcome {username}\nSetting up your profile...")
        user = User(username)
        role_selection_menu(user)
        # Returns here when user exits role_selection_menu
    else:
        print("\nUsername not found.")


def main():
    login_options = ("login", "new user", "exit")
    while True:
        login_option = get_option(login_options, "login option")

        if login_option == "login":
            handle_login()
        elif login_option == "new user":
            newUser()
        elif login_option == "exit":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()