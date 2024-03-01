from typing import List


def handle_args_to_ignore(args: List[str]):
    indices_to_remove = []
    for i, arg in enumerate(args):
        if "_ignore_" in arg:
            indices_to_remove.append(i)
            if not arg.startswith("-"):
                indices_to_remove.append(i - 1)

    for i in sorted(indices_to_remove, reverse=True):
        del args[i]
