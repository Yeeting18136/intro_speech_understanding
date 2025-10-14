# Lecture 2

Here's the code for lecture 2.  Fetch and merge this to your laptop:
```
git fetch release
git merge release/main -m "Merging release" --allow-unrelated-histories
```
Edit `homework2.py`, then check to make sure it worked by typing
```
python grade.py
```
It will print out some commentary, and a grade.  If you like your grade, upload your code to your github repository, so that your instructor can see it:

1. <a href="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token">Create a fine-grained personal access token</a> that times out after 24 hours, and that only has access to your intro_speech_understanding repository.

2. Use the following commands to upload your new code.  After `git push`, you'll be asked for your github username and a password.  In place of the password, use the peronal access token.

```
git add .
git commit -m "hw1"
git push origin main
```

