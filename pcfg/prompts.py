def get_prompts(functions):
    if functions == "shift_repeat_cut":
        fwd_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return [x[-1]] + x[:-1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
"""

        back_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the backwards sequence of transformations.

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return [x[-1]] + x[:-1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
"""

        flip_prompt_header = fwd_prompt_header

        verify_prompt_fixed = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return [x[-1]] + x[:-1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
Initial: [3, 0, 8, 8, 3, 0, 8, 8]
Desired Final: [8, 8, 0, 3]
Functions: [shift_left, cut, shift_right]
Verify initial to final steps:
  shift_left: [3, 0, 8, 8, 3, 0, 8, 8][1:] + [3, 0, 8, 8, 3, 0, 8, 8][:1] -> [0, 8, 8, 3, 0, 8, 8, 3]
  cut: [0, 8, 8, 3, 0, 8, 8, 3] half -> [0, 8, 8, 3] and [0, 8, 8, 3] equal -> [0, 8, 8, 3]
  shift_right: [[0, 8, 8, 3][-1]] + [0, 8, 8, 3][:-1] -> [3, 0, 8, 8]
  actual final: [3, 0, 8, 8], desired final: [8, 8, 0, 3], does not match
  Incorrect

Initial: [1, 5, 7, 2]
Desired Final: [7, 2, 1, 5, 7, 2, 1, 5]
Functions: [shift_right, shift_right, repeat]
Verify initial to final steps:
  shift_right: [[1, 5, 7, 2][-1]] + [1, 5, 7, 2][:-1] -> [2, 1, 5, 7]
  shift_right: [[2, 1, 5, 7][-1]] + [2, 1, 5, 7][:-1] -> [7, 2, 1, 5]
  repeat: [7, 2, 1, 5] + [7, 2, 1, 5] -> [7, 2, 1, 5, 7, 2, 1, 5]
  actual final: [7, 2, 1, 5, 7, 2, 1, 5], desired final: [7, 2, 1, 5, 7, 2, 1, 5], match
  Correct

Initial: [5, 5, 0, 2, 5, 5, 3, 2]
Desired Final: [4, 2, 0, 5, 5]
Functions: [shift_left, cut, shift_right]
Verify initial to final steps:
  shift_left: [5, 5, 0, 2, 5, 5, 3, 2][1:] + [5, 5, 0, 2, 5, 5, 3, 2][:1] -> [5, 0, 2, 5, 5, 3, 2, 5]
  cut: [5, 0, 2, 5, 5, 3, 2, 5] half -> [5, 0, 2, 5] and [5, 3, 2, 5] not equal -> cut failed
  Incorrect

Initial: [2, 5, 9, 5]
Desired Final: [2, 5, 9, 5, 2, 5, 9, 5]
Functions: [shift_right, repeat, shift_left]
Verify initial to final steps:
  shift_right: [[2, 5, 9, 5][-1]] + [2, 5, 9, 5][:-1] -> [5, 2, 5, 9]
  repeat: [5, 2, 5, 9] + [5, 2, 5, 9] -> [5, 2, 5, 9, 5, 2, 5, 9]
  shift_left: [5, 2, 5, 9, 5, 2, 5, 9][1:] + [5, 2, 5, 9, 5, 2, 5, 9][:1] -> [2, 5, 9, 5, 2, 5, 9, 5]
  actual final: [2, 5, 9, 5, 2, 5, 9, 5], desired final: [2, 5, 9, 5, 2, 5, 9, 5], match
  Correct
"""

    elif functions == "repeat_cut_reverse_swap":
        fwd_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
"""

        back_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the backwards sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
"""

        flip_prompt_header = fwd_prompt_header

        verify_prompt_fixed = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

def repeat(x):
  # repeat the sequence once
  return x + x

def cut(x):
  # cut the sequence in half
  assert x[:len(x) // 2] == x[len(x) // 2:]
  return x[:len(x) // 2]

***** Examples:
Initial: [3, 8, 8, 3, 0, 8, 8, 0]
Desired Final: [8, 8, 0, 3]
Functions: [swap, cut, reverse]
Verify initial to final steps:
  swap: [3, 8, 8, 3, 0, 8, 8, 0][-1:] + [3, 8, 8, 3, 0, 8, 8, 0][1:-1] + [0, 8, 8, 3, 6][:1] -> [0, 8, 8, 3, 0, 8, 8, 3]
  cut: [0, 8, 8, 3, 0, 8, 8, 3] half -> [0, 8, 8, 3] and [0, 8, 8, 3] equal -> [0, 8, 8, 3]
  reverse: [0, 8, 8, 3][::-1] -> [3, 8, 8, 0]
  actual final: [3, 8, 8, 0], desired final: [8, 8, 0, 3], does not match
  Incorrect

Initial: [1, 5, 7, 2]
Desired Final: [1, 7, 5, 2, 1, 7, 5, 2]
Functions: [reverse, swap, repeat]
Verify initial to final steps:
  reverse: [1, 5, 7, 2][::-1] -> [2, 7, 5, 1]
  swap: [2, 7, 5, 1][-1:] + [2, 7, 5, 1][1:-1] + [2, 7, 5, 1][:1] -> [1, 7, 5, 2]
  repeat: [1, 7, 5, 2] + [1, 7, 5, 2] -> [1, 7, 5, 2, 1, 7, 5, 2]
  actual final: [1, 7, 5, 2, 1, 7, 5, 2], desired final: [1, 7, 5, 2, 1, 7, 5, 2], match
  Correct

Initial: [5, 5, 0, 2, 5, 5, 3, 2]
Desired Final: [4, 2, 0, 5, 5]
Functions: [reverse, cut, swap]
Verify initial to final steps:
  reverse: [5, 5, 0, 2, 5, 5, 3, 2][::-1] -> [2, 3, 5, 5, 2, 0, 5, 5]
  cut: [2, 3, 5, 5, 2, 0, 5, 5] half -> [2, 3, 5, 5] and [2, 0, 5, 5] not equal -> cut failed
  Incorrect

Initial: [2, 5, 9, 5]
Desired Final: [2, 9, 5, 2, 5, 9, 5, 5]
Functions: [reverse, repeat, swap]
Verify initial to final steps:
  reverse: [2, 5, 9, 5][::-1] -> [5, 9, 5, 2]
  repeat: [5, 9, 5, 2] + [5, 9, 5, 2] -> [5, 9, 5, 2, 5, 9, 5, 2]
  swap: [5, 9, 5, 2, 5, 9, 5, 2][-1:] + [5, 9, 5, 2, 5, 9, 5, 2][1:-1] + [5, 9, 5, 2, 5, 9, 5, 2][:1] -> [2, 9, 5, 2, 5, 9, 5, 5]
  actual final: [2, 9, 5, 2, 5, 9, 5, 5], desired final: [2, 9, 5, 2, 5, 9, 5, 5], match
  Correct
"""

    elif functions == "shift_reverse_swap":
        fwd_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return [x[-1]] + x[:-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

***** Examples:
"""

        back_prompt_header = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the backwards sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return [x[-1]] + x[:-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

***** Examples:
"""

        flip_prompt_header = fwd_prompt_header

        verify_prompt_fixed = """A random sequence of three of the below functions transform the initial array into the final array.
Given initial and final, output the sequence of transformations.

def reverse(x):
  # reverse the sequence
  return x[::-1]

def shift_left(x):
  # shift the sequence to the left by one
  return x[1:] + x[:1]

def shift_right(x):
  # shift the sequence to the right by one
  return x[-1:] + x[:-1]

def swap(x):
  # swap the first and last elements
  return x[-1:] + x[1:-1] + x[0:1]

***** Examples:
Initial: [1, 5, 7, 1, 2]
Desired Final: [1, 7, 5, 2, 1]
Functions: [reverse, swap, shift_left]
Verify initial to final steps:
  reverse: [1, 5, 7, 1, 2][::-1] -> [2, 1, 7, 5, 1]
  swap: [2, 1, 7, 5, 1][-1:] + [2, 1, 7, 5, 1][1:-1] + [2, 1, 7, 5, 1][:1] -> [1, 1, 7, 5, 2]
  shift_left: [1, 1, 7, 5, 2][1:] + [1, 1, 7, 5, 2][:1] -> [1, 7, 5, 2, 1]
  actual final: [1, 7, 5, 2, 1], desired final: [1, 7, 5, 2, 1], match
  Correct

Initial: [0, 8, 8, 3, 6]
Desired Final: [3, 0, 8, 8, 6]
Functions: [shift_right, swap, reverse]
Verify initial to final steps:
  shift_right: [0, 8, 8, 3, 6][-1:] + [0, 8, 8, 3, 6][:-1] -> [6, 0, 8, 8, 3]
  swap: [6, 0, 8, 8, 3][-1:] + [6, 0, 8, 8, 3][1:-1] + [6, 0, 8, 8, 3][:1] -> [3, 0, 8, 8, 6]
  reverse: [3, 0, 8, 8, 6][::-1] -> [6, 8, 8, 0, 3]
  actual final: [6, 8, 8, 0, 3], desired final: [3, 0, 8, 8, 6], does not match
  Incorrect

Initial: [5, 5, 0, 2, 4]
Desired Final: [4, 2, 0, 5, 5]
Functions: [swap, reverse, swap]
Verify initial to final steps:
  swap: [5, 5, 0, 2, 4][-1:] + [5, 5, 0, 2, 4][1:-1] + [5, 5, 0, 2, 4][:1] -> [4, 5, 0, 2, 5]
  reverse: [4, 5, 0, 2, 5][::-1] -> [5, 2, 0, 5, 4]
  swap: [5, 2, 0, 5, 4][-1:] + [5, 2, 0, 5, 4][1:-1] + [5, 2, 0, 5, 4][:1] -> [4, 2, 0, 5, 5]
  actual final: [4, 2, 0, 5, 5], desired final: [4, 2, 0, 5, 5], match
  Correct

Initial: [2, 5, 9, 1, 5]
Desired Final: [5, 5, 9, 1, 2]
Functions: [shift_left, reverse, shift_right, swap]
Verify initial to final steps:
  shift_left: [2, 5, 9, 1, 5][1:] + [2, 5, 9, 1, 5][:1] -> [5, 9, 1, 5, 2]
  reverse: [5, 9, 1, 5, 2][::-1] -> [2, 5, 1, 9, 5]
  shift_right: [2, 5, 1, 9, 5][-1:] + [2, 5, 1, 9, 5][:-1] -> [5, 2, 5, 1, 9]
  swap: [5, 2, 5, 1, 9][-1:] + [5, 2, 5, 1, 9][1:-1] + [5, 2, 5, 1, 9][:1] -> [9, 2, 5, 1, 5]
  actual final: [9, 2, 5, 1, 5], desired final: [5, 5, 9, 1, 2], does not match
  Incorrect
"""

    return (
        fwd_prompt_header,
        back_prompt_header,
        flip_prompt_header,
        verify_prompt_fixed,
    )
