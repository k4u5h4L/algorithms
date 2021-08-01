import sys
import os

content = f"""
# {sys.argv[2]} programs:

"""

file_names = os.listdir(sys.argv[2])

for file_name in file_names:
    with open(f"{sys.argv[2]}/{file_name}", "r") as f:
        content += "```py\n"
        content += f.read()
        content += "\n```\n"

        f.close()

with open(f"{sys.argv[2].upper()}.md", "w") as f:
    f.write(content)
    f.close()

print("Content has been written to file.")
