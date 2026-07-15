from git import Repo

repo = Repo(search_parent_directories=True)
print(repo.head.object.hexsha)