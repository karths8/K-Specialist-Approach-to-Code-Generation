import json
import random
import re

TRANSITIONS_TO_WEIGHT = [
    (">>> {func_name}({args})\n{expected_return}", 1/12, 3),
    ("{func_name}({args}) => {expected_return}", 1/12, 3),
    ("{func_name}({args}) ==> {expected_return}", 1/12, 3),
    ("{func_name}({args}) -> {expected_return}", 1/12, 3),
    ("{func_name}({args}) ➞ {expected_return}", 1/12, 3),
    ("Input: {args}\nOutput: {expected_return}", 1/12, 2),
    ("{func_name}({args}) == {expected_return}", 1/12, 3),
    ("{func_name}({args}) = {expected_return}", 1/12, 3),
    ("For input = {args} the output should be {expected_return}.", 1/12, 2),
    ("{func_name}({args})   # returns {expected_return}", 1/12, 3),
    ("{func_name}({args}) returns {expected_return}", 1/12, 3),
    ("{func_name}({args}) should return {expected_return}", 1/12, 3)
]
def read_human_eval(file):
    human_eval_data = []
    with open(file, 'r') as fp:
        human_eval_data = json.load(fp)
    for ex in human_eval_data:
        print(ex['question'])

def find_closing_parenthesis(script: str, opening_parenthesis_index: int):
    stack = []
    for i in range(opening_parenthesis_index, len(script)):
        if script[i] == '(':
            stack.append('(')
        elif script[i] == ')':
            if not stack:
                return -1  # No matching opening parenthesis found
            stack.pop()
            if not stack:
                return i  # Found the closing parenthesis
    return -1  # No closing parenthesis found

def _parse_asserts(asserts: str):
    ret = []
    last_idx = 0
    cur_match = re.search(r'assert\s*\({0,1}\s*', asserts[last_idx:])
    while last_idx < len(asserts) and cur_match:
        func_sig_start = cur_match.start() + last_idx
        last_idx = cur_match.end() + last_idx
        func_name_end_ = re.search(r'[\s(]', asserts[last_idx:])
        if func_name_end_:
            func_name = asserts[last_idx:func_name_end_.start()+ last_idx]
            close_par = find_closing_parenthesis(asserts, func_name_end_.start()+ last_idx)
            if close_par >= 0:
                args = asserts[func_name_end_.start()+ last_idx + 1:close_par]
                equal_match = re.search(r'\s*(<|<=|==|>|>=)\s*', asserts[close_par:])
                if equal_match is not None:
                    excpected_return_start = close_par + equal_match.end()
                    excpected_return_end = max(asserts.find('\n', excpected_return_start), excpected_return_start+1)
                    expected_return = asserts[excpected_return_start:excpected_return_end]
                else:
                    expected_return = 'True'
                ret.append((func_name, args, expected_return))
                #func_signatures.append(func_sig)
        cur_match = re.search(r'assert\s*\({0,1}\s*', asserts[last_idx:])
    return ret

def transform_asserts(assert_list):
    if type(assert_list) is str:
        assert_list = [assert_list]
    ret = []
    transition_weights = [t[1] for t in TRANSITIONS_TO_WEIGHT]
    idxs = list(range(len(transition_weights)))
    for item in assert_list:
        item_ret = []
        transition_idx = random.choices(idxs, transition_weights)
        transition = TRANSITIONS_TO_WEIGHT[transition_idx[0]][0]
        num_args = TRANSITIONS_TO_WEIGHT[transition_idx[0]][2]
        parsed_asserts = _parse_asserts(item)
        for func_args_ret in parsed_asserts:
            if num_args == 2:
                item_ret.append(transition.format(args = func_args_ret[1], expected_return = func_args_ret[2]))
            elif num_args == 3:
                item_ret.append(transition.format(func_name = func_args_ret[0], args = func_args_ret[1], expected_return = func_args_ret[2]))
        ret.append(item_ret)
    return ret

if __name__=="__main__":
    with open('Notebooks/xlcost_filtered_results_Dec12.json', 'r') as fp:
      data = json.load(fp)
      ret = transform_asserts([row['asserts'] for row in data])
      [print(r) for r in ret]