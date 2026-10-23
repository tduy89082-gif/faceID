from database import (
    init_db,
    get_all_users,
    get_all_attendance
)

from register import (
    register_user
)

from main import (
    recognize
)


def show_users():

    users = get_all_users()

    print()
    print("========== USERS ==========")

    if not users:

        print("No users")

        return


    for user in users:

        print(
            f"{user['id']} - "
            f"{user['name']} - "
            f"{user['created_at']}"
        )


def show_attendance():

    records = get_all_attendance()

    print()
    print("======= ATTENDANCE ========")

    if not records:

        print("No attendance records")

        return


    for record in records:

        print(
            f"{record['user_id']} - "
            f"{record['name']} - "
            f"{record['timestamp']}"
        )


def main():

    init_db()


    while True:

        print()
        print("==========================")
        print("       FACE ID SYSTEM")
        print("==========================")

        print("1. Register User")
        print("2. Start Face Recognition")
        print("3. View Users")
        print("4. View Attendance")
        print("5. Exit")


        choice = input(
            "Choose: "
        ).strip()


        if choice == "1":

            register_user()


        elif choice == "2":

            recognize()


        elif choice == "3":

            show_users()


        elif choice == "4":

            show_attendance()


        elif choice == "5":

            print("Goodbye")

            break


        else:

            print(
                "Invalid choice"
            )


if __name__ == "__main__":

    main()