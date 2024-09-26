import re

def remove_docstrings(filename):
    with open(filename, 'r') as file:
        content = file.read()
        
    # Regular expression to match docstrings
    pattern = r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'|"(?:.|\n)*?"|\'(?:.|\n)*?\''
    
    # Remove docstrings from content
    new_content = re.sub(pattern, '', content)
    
    with open(filename, 'w') as file:
        file.write(new_content)

# Usage example
file = "~/Documents/phd/projs/demerror/demcompare/rootaggregatesquarederror/root_agg_squared_error.py"
remove_docstrings(file)


