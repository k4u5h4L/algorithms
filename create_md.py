import sys
import os

def chunker(seq, size):
    return list(seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_init_content(page):
    return f"""
# {sys.argv[2]} programs:
## Page: {page}
"""
    

content = get_init_content(1)

file_names = os.listdir(sys.argv[2])

programs_per_page = 30
page = 1

file_chunks = chunker(file_names, programs_per_page)
path = f'./docs/{sys.argv[2]}'

try:  
    os.mkdir(path)  
except FileExistsError as error:  
    print('Folder already exists. Just gonna use that folder.')
    print(error)

for chunk in file_chunks:
    for file_name in chunk:
        with open(f"{sys.argv[2]}/{file_name}", "r") as f:
            temp = file_name[:-3]
            temp = temp.split("_")
            temp = " ".join(temp)
            content += f"## {temp}\n\n"
            t = file_name.split(".")[-1]
            content += f"```{t}\n"
            content += f.read()
            content += "\n```\n\n"

    with open(f"{path}/{sys.argv[2].upper()}-{page}.md", "w") as f:
        f.write(content)
    
    page += 1
    content = get_init_content(page)

print("Content has been written to file(s).")
