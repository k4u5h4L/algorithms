import sys
import os

content = f"""
# {sys.argv[2]} programs:

"""

file_names = os.listdir(sys.argv[2])

for file_name in file_names:
    with open(f"{sys.argv[2]}/{file_name}", "r") as f:
        temp = file_name[:-3]
        temp = temp.split("_")
        temp = " ".join(temp)
        content += f"## {temp}\n\n"
        t = file_name.split(".")[-1]
        content += f"```{t}\n"
        content += f.read()
        content += "\n```\n\n"

with open(f"{sys.argv[2].upper()}.md", "w") as f:
    f.write(content)

print("Content has been written to file.")
