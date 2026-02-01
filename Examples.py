even = 0
for i in range(1, 10):  # range excludes the last number
    if i % 2 == 0:
        print(i, "even")
        even += 1
print(f"Total even numbers between 1 and 10: {even}")


def greet(first_name, last_name=""):  # function def. function has parameters
    print("Hello!")
    print(f"welcome {first_name} {last_name}")
    print("Welcome to the program.\n")
    return f"Greeted {first_name} {last_name}"


first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")

greet(first_name, last_name)  # function call. function has arguments
greet("Aarya", "Cat")
greet("kumar")  # last_name will use default value

# all functions return None if there is no return statement


def multiply(*numbers):  # variable number of arguments
    result = 1
    for number in numbers:
        result *= number
    return result


print(multiply(2, 3, 4, 5))
