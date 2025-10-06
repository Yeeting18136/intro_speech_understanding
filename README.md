# intro_speech_understanding

This repository contains assignments for the <a
href="https://www.kcg.edu/">KCGI</a> graduate course "Intro to Speech
Understanding."

---------------------------------------------------------------

# Getting Started

1. <a href="https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github">Create your own account on github.com</a>
2. <a href="https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo">Fork this repository</a> in your own account.
3. Open a terminal on your laptop:
    1. <a href="https://learn.microsoft.com/en-us/windows/terminal/install">Windows</a>
    1. <a href="https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac">MacOS</a>
    1. <a href="https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal">Linux</a>
4. <a href="https://github.com/git-guides/install-git">Install git on your laptop.</a>
5. <a href="https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository">Clone your fork</a> onto your laptop by going to the terminal and typing
```
git clone https://github.com/<your_github_account_name>/intro_speech_understanding
```
6. <a href="https://docs.github.com/en/get-started/git-basics/managing-remote-repositories">Define two remotes</a>.  One remote should be called `origin`, and should point to your own github fork. One should be called `release`, and should point to my fork.  If these don't already exist, you can create them by typing:
```
git remote add release https://github.com/jhasegaw/intro_speech_understanding
git remote add origin https://github.com/<your_github_account_name>/intro_speech_understanding
```
Very likely, the second command will cause an error: `fatal: remote origin already exists`.  That's fine!  It means that `origin` already exists.  To make sure, type
```
git remote -v
```
It should now report that both `release` and `origin` exist.

7. <a href="https://www.anaconda.com/">Install python on your laptop.</a>

8. <a href="https://github.com/settings/personal-access-tokens">Create a personal access token for this repository,</a> and save it somewhere on your laptop, so you can use it to publish code.

---------------------------------------------------------------

# Weekly Update

This repository will change every week.  In order to merge its changes
into the copy on your laptop:

1. Make sure to <a href="https://git-scm.com/docs/git-commit">save your local changes:</a>
```
git commit -am "saving changes"
```
2. <a href="https://git-scm.com/docs/git-fetch">Fetch the release, and <a href="https://git-scm.com/docs/git-merge">merge</a> it with your local copy:
```
git fetch release
git merge release/main -m "Merging release" --allow-unrelated-histories
```
3. Do the homework
4. Save your changes, then push them to your github fork:
```
git add .
git commit -m "saving changes"
git push origin main
```
You will need to log in using your github account name.  Instead of a password, use your <a href="https://github.com/settings/personal-access-tokens">personal access token.</a>

#### If git push has an error
If you've made any changes on your github fork, your `git push` command will fail.  In that case, you should first merge the changes into the local copy on your laptop:
```
git fetch origin
git merge origin/main -m "Merging github changes" --allow-unrelated-histories
```
Edit your local files, to make sure that the result of `git merge` is what you wanted.  Then try again:
```
git add .
git commit -m "saving changes"
git push origin main
```


## Disaster Recovery

When you are new to git, you will often find that git is asking you a
question that you don't know how to answer.

If that happens, the easiest solution is to just start over!  Don't
delete your existing local copy.  Instead, create a new local copy.
The total amount of space required by this repository is tiny; it's
totally fine for you to have many copies of it.

