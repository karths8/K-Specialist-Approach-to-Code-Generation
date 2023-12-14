import json
import re

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
  func_names = set()
  last_idx = 0
  cur_match = re.search(r'assert\s*\({0,1}\s*', asserts[last_idx:])
  while last_idx < len(asserts) and cur_match:
    last_idx = cur_match.end() + last_idx
    func_end_ = re.search(r'[\s(]', asserts[last_idx:])
    if func_end_:
      func_names.add(asserts[last_idx:func_end_.start()+ last_idx])
    cur_match = re.search(r'assert\s*\({0,1}\s*', asserts[last_idx:])
  return func_names

def _parse_code2(code: str, asserts: str):
  imports = []
  func_signatures = []
  lines = code.splitlines()
  idx = 0
  while idx < len(lines):
    if ' import ' in lines[idx] or lines[idx][:7] == 'import ' or ' from ' in lines[idx] or lines[idx][:5] == 'from ':
      import_str = lines[idx]
      while import_str[-1] == "\\": #Handle newline 
        idx += 1
        import_str += '\n' + lines[idx]
      imports.append(import_str)
    idx+=1

  #search functions
  last_idx = 0
  cur_match = re.search(r'[\t ]{0,1}def\s+', code[last_idx:])
  while last_idx < len(code) and cur_match:
    func_sig_start = cur_match.start() + last_idx
    last_idx = cur_match.end() + last_idx
    func_name_end_ = re.search(r'[\s(]', code[last_idx:])
    if func_name_end_:
      func_name = code[last_idx:func_name_end_.start()+ last_idx]
      match_in_assert = re.search(r'\s*[\s(\.]' + func_name + r'[\s(]', asserts)
      if match_in_assert:
        close_par = find_closing_parenthesis(code, func_name_end_.start()+ last_idx)
        func_sig_end = code.find(':', close_par)
        func_sig = code[code.find('def',func_sig_start):func_sig_end+1]
        print(func_sig)
        func_signatures.append(func_sig)
    cur_match = re.search(r'\s*def\s+', code[last_idx:])

  return {"function_signature": func_signatures, "imports": imports}

  #\s+[a-zA-Z_][a-zA-Z0-9]*
def _parse_code(code: str, func_names):
  imports = []
  func_signatures = []
  lines = code.splitlines()
  idx = 0
  while idx < len(lines):
    if ' import ' in lines[idx] or lines[idx][:7] == 'import ' or ' from ' in lines[idx] or lines[idx][:5] == 'from ':
      import_str = lines[idx]
      while import_str[-1] == "\\": #Handle newline 
        idx += 1
        import_str += '\n' + lines[idx]
      imports.append(import_str)
    idx+=1

  for func_name in func_names:
    sub_func_idx = func_name.rfind('.')
    if sub_func_idx >= 0:
      func_name = func_name[sub_func_idx+1:]

    match = re.search(r'def\s+' + func_name, code)
    if match:
      close_par = find_closing_parenthesis(code, match.end())
      func_sig_end = code.find(':', close_par)
      func_sig = code[match.start():func_sig_end+1]
      print(func_sig)
      func_signatures.append(func_sig)
  
  return {"function_signature": func_signatures, "imports": imports}

def to_import_statements_func_sig(json_file: str):
  data = None
  out = []
  with open(json_file, 'r') as fp:
    data = json.load(fp)
  for row in data:
    #func_names =  _parse_asserts(row['asserts'])
    #out.append(_parse_code(,func_names))
    out.append(_parse_code2(row['code'], row['asserts']))
  return out

if __name__=="__main__":
  ret = to_import_statements_func_sig('Notebooks/xlcost_filtered_results_Dec12.json')