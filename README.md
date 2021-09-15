# Random Algorithms in Python

Just solutions to random problems I find. There are also about 200+ leetcode solutions here as well. Check the `leetcode/` dir for the problems.


### Find the number of problems in each folder:

```bash
python check-no.py --folder <folder_name>

# eg-

python check-no.py --folder leetcode
```

This will output the number of problems in the `leetcode/` folder.


### Make a `.md` file based on all the programs in a folder

```bash
python create_md.py --folder <folder_name>

# eg-

python create_md.py --folder leetcode
```

This will make a new file called `<folder_name_caps>.md` with all the programs in the folder which makes it easier to read.

## Note:

- Before committing, run the `python check-no.py --folder <folder_name>` command in whatever folder you have updated to generate an `md` file so its easier to read.
- Please do not upload code which is incorrect. If you know its incorrect but it still has value, put them in the folder `not_sure_about_these/` so that the reader knows it may contain a mistake. 
- While uploading code snippets from different sites such as Leetcode, Hackerrank, Hackerearth; please paste the question/problem statement before your solution in comments. It helps to find the solution better and easier to read.
