import math

def log_message(level:str, fnc_name, start=True):
	level = level.split('.')
	indent = level[0]
	indent = int(indent)
	indent = indent - 1
	indent = '    ' * indent
	level = '.'.join(level)

	print("\n" + "="*80)
	print(f"\033[1m{level}: {indent}{fnc_name}\033[0m", end="")
	if start:
		print(" -> S T A R T E D")
	else:
		print(" -> F I N I S H E D")
	print()


import os
import inspect

def get_current_function_name(stack_level=1):
    return inspect.stack()[stack_level].function

def get_current_class_name(stack_level=1):
    frame = inspect.stack()[stack_level].frame
    while frame:
        if 'self' in frame.f_locals:
            return frame.f_locals['self'].__class__.__name__
        frame = frame.f_back
    return None



import os
import inspect

def get_class_name_from_frame(frame):
    if 'self' in frame.f_locals:
        return frame.f_locals['self'].__class__.__name__
    return None

def print_project_details():
    print()
    stack = inspect.stack()
    call_chain = []

    for frame_info in stack[1:]:  # Skip the first frame (print_project_details itself)
        file_name = os.path.basename(frame_info.filename)
        function_name = frame_info.function
        class_name = get_class_name_from_frame(frame_info.frame)

        if class_name:
            call_chain.append(f"{file_name}.{class_name}().{function_name}")
        else:
            call_chain.append(f"{file_name}.{function_name}")

    # Join the call chain into a single string
    call_chain_str = " -> ".join(call_chain)
    print(call_chain_str)
    print()