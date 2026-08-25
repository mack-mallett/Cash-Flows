from appLogic.users import User
import appLogic.newUser as newUser
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
        user_option = input('Option: ')
        if user_option.lower() in available_options:
            return user_option.lower()  
        else:
            print()
            print('function fail')


def main():
    while True:
        login_options = ('login', 'new user')
        login_option = get_option(login_options, 'login_option.')

        if login_option == 'login':
            while True:
                print()
                username = input('Enter Username: ')
                usernames = pd.read_csv(Path.cwd() / 'userNames.csv')

                if username in usernames['User Names'].values:
                    print()
                    print(f'Welcome {username}')
                    print('Setting up your profile...')
                    #alternativly this could call a subscript, which launches the program by creating a user. This would end the While Loop.
                    user = User(username)
                else:
                    menu_else()
                while True:
                    role_options = ('admin', 'analysis')
                    role_option = get_option(role_options, 'group of tasks.')
                    if role_option == 'admin':
                        while True:
                            admin_options = ('add purchases'
                                                ,'update budget'
                                                ,'update your jobs'
                                                ,'exit to Admin Tasks')
                            admin_option = get_option(admin_options, 'pocket book function.').lower()

                            if admin_option == 'add purchases':
                                while True:
                                    print()
                                    print('Have you Confirmed that all spreadsheets are filled in properly,',
                                          ' and are in the right place?')
                                    user_inputs = ('yes', 'no')
                                    user_input = get_option(user_inputs, 'yes or no')
                                    if user_input == 'yes':
                                        user.updateAccounts()
                                        print('Your General Ledger has been updated with the latest information.')
                                        break
                                    elif user_input == 'no':
                                        print()
                                        print('Please make adjustments and come back.')
                                        break
                                    else:
                                        menu_else()           
                            elif admin_option == 'update budget':
                                print()
                                print('Create a module for updating the budget\nMake sure it can connect to the Analysis Module')
                            elif admin_option == 'update your jobs':
                                print()
                                print('Create a module for updating the jobs')
                            elif admin_option == 'exit to admin tasks':
                                break
                            else:
                                menu_else()
                    elif role_option == 'analysis':
                        print()
                        print('Create an Analysis Module. Start with cash flows!')
                    else:
                        menu_else()
        elif login_option == 'new user':
            print()
            print('Connect me to the new user script!')

            #run new user script
        else:
            menu_else()

if __name__ == '__main__':
    main()