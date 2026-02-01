import os
import math


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_screen()

# \'
# \n

print("Hello, World!")

students_count = 100
rating = 4.5
is_published = False
course_name = "Learning Python"
print(students_count)
print(course_name)

message = """
Aarya is a cat.
multi line string
Bye
"""

print(message)

length = len(course_name)
print(length)
print(course_name[0])  # 'L'
print(course_name[-1])  # 'n'
print(course_name[0:7])  # 'Learning'
print(course_name[0:])  # 'Learning Python'

first = "kumar"
last = "gaurav"
full = first + " " + last
print(full)
full = f"{first} {last}"
print(full)
full = f"{len(first)} {last} is learning Python {2+2}"
print(full)

course = "    Python Programming"
print(course.upper())
print(course.lower())
print(course.title())
print(course.strip())  # Remove leading/trailing whitespace
print(course.find("Pro"))  # Find the index of "Pro"
print(course.replace("P", "J"))  # Replace "P" with "J"
print("Python" in course)  # Check if "Python" is in the course string
print("Java" not in course)  # Check if "Java" is in the course string

# 3 types of number. Int, float, complex
x = 10  # int
y = 3.14  # float
z = 2 + 3j  # complex
print(10+3)
print(10 / 3)
print(10 // 3)  # Floor division
print(10 % 3)  # Modulus
print(10 ** 3)  # Exponentiation

x += 5  # x = x + 5
x *= 2  # x = x * 2
print(x)

print(round(2.9))

print(math.ceil(2.9))

name = input("Enter your name: ")
print(f"Hello, {name}!")
age = int(input("Enter your age: "))
print(f"You are {age} years old {name}.")

# Falsy values in Python: 0, 0.0, "", None, [], {}, set()

print(bool(0))  # False
print(bool(25))  # True
print(bool(""))  # False
print(bool("Hello"))  # True
print(bool("False"))  # True
print(bool("True"))  # True

temperature = 25
if temperature > 30:
    print("It's a hot day")
elif temperature > 20:
    print("It's a nice day")
else:
    print("It's a cold day")

age = 22
if age >= 18:
    message = "Eligible to vote"
else:
    message = "Not eligible"

print(message)

message = "Eligible to vote" if age >= 18 else "Not eligible"
print(message)

# logical operators: and, or, not
high_income = True
good_credit = True

if high_income and good_credit:
    print("Eligible for loan")
else:
    print("Not eligible for loan")

# age between 18 and 65
# chain comparison operators
age = 25
if age >= 18 and age <= 65:
    print("Eligible for work")

if 18 <= age <= 65:
    print("Eligible for work")

# for loop
print("For Loop:")
for i in range(5):
    print("Attempt", i+1, (i+1)*".")

for i in range(1, 4, 1):
    print("Attempt", i, (i)*".")

successful = True
for i in range(5):
    print("Attempt", i+1)
    if successful:
        print("Successful")
        break
else:
    print("Attempted 5 times and failed")

# nested loops
print("Nested Loops:")
for x in range(3):
    for y in range(3):
        print(f"({x}, {y})")

print(type(5))  # <class 'int'>
print(type(range(5)))  # <class 'range'>
print(type("Hello"))  # <class 'str'>

for x in "Kumar":
    print(x)

for x in [1, 2, 3]:
    print(x)
# while loop
print("While Loop:")
i = 0
while i < 5:
    print("Attempt", i+1)
    i += 1

# infinite loop
# while True:
#     print("Hello World")
#     break
print("Done")

# break, continue, pass
print("Break, Continue, Pass:")

print(4 % 2)  # 0
