# Contributing to Portfolio Backtester

First off, thank you for considering contributing to Portfolio Backtester! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/your-username/portfolio-backtester/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If this is something you think you can fix, then [fork Portfolio Backtester](https://github.com/your-username/portfolio-backtester/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```bash
git checkout -b 33-add-new-optimization-metric
```

### Get the code

```bash
# Clone your fork to your local machine
git clone https://github.com/your-username/portfolio-backtester.git

# Go to the project directory
cd portfolio-backtester

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Run the tests

To make sure everything is working as expected, run the tests:

```bash
python -m pytest
```

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :)

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Portfolio Backtester's master branch:

```bash
git remote add upstream https://github.com/your-username/portfolio-backtester.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```bash
git checkout 33-add-new-optimization-metric
git rebase master
git push --set-upstream origin 33-add-new-optimization-metric
```

Finally, go to GitHub and [make a Pull Request](https://github.com/your-username/portfolio-backtester/compare) :D

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing and merging, check out this guide: [About Git rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase).
