"""
string utils

"""


def flip_numbers_in_string(numbers_string):
    numbers_string = numbers_string[1:-1]  # Remove the brackets
    # Split the string by commas and strip any whitespace
    numbers_list = [num.strip() for num in numbers_string.split(",")]
    # Reverse the order of the elements
    reversed_numbers_list = list(reversed(numbers_list))
    # Join the elements back into a string with commas and restore the bracket structure
    reversed_string = "(" + ", ".join(reversed_numbers_list) + ")"
    return reversed_string


# def flip_numbers_in_string(numbers_string):
#     # Split the string by commas and strip any whitespace
#     numbers_list = [num.strip() for num in numbers_string.split('<-')]
#     # Reverse the order of the elements
#     reversed_numbers_list = list(reversed(numbers_list))
#     # Join the elements back into a string with commas and restore the bracket structure
#     reversed_string = ' -> '.join(reversed_numbers_list)
#     return reversed_string

if __name__ == "__main__":
    numbers_string = "(7, 9, 10)"
    # numbers_string = "7 <- 9 <- 10"
    reversed_string = flip_numbers_in_string(numbers_string)
    print(reversed_string)
