
# HW4: Applying Trajectory Optimization 
## Due date: Wednesday, April 21

In this homework you'll implement a few of the methods we've talked about in class that extend the traditional formulation of trajectory optimization.
1. Design a barrell roll maneuver for an airplane by modifying iLQR to work with quaternions
2. Design a walking gain for a simple bipedal walker using Hybrid Trajectory Optimization (using Ipopt)
3. Refine an optimized solution to compensate for modelling errors using Iterative Learning Control (ILC)

## Running the Autograder
The autograder setup has changed slightly for this assignment. The tests are now included in the notebooks themselves. Please do not modify the tests! We did this to make it easier for you to 
get quick feedback. Also, do not assume that just because your solution passes the tests it will pass all future tests. They are meant to help guide you in your implementation, 
not as a definitive catch-all for all errors that might occur. You should get in the habit of designing tests like this as your write your own programs, especially in your projects!

1. Open a terminal in the root directory of your repo
2. Launch a Julia REPL
3. Enter the package manager using `]` and enter `activate .`
4. Launch the testing suite using `test hw4`

## Submitting your homework
Make sure your repo lives under the Class Organization. This will be done automatically when you use the GitHub Classrooms link we send provide.

1. Add your info to the `studentinfo` function in [src/hw4.jl](https://github.com/Optimal-Control-16-745/hw3/blob/main/src/hw4.jl)
2. Create a release. Follow [these instructions](https://github.com/Optimal-Control-16-745/JuliaIntro/blob/main/docs/Submission%20Instructions.md) for more info on creating the release.

## Adding the Upstream Repo
We may release changes to the homework periodically if errors or bugs are found. Follow these instructions for linking your repo to the original template and pulling changes. It's always a good idea to branch your code before pulling from the upstream repo in case something goes wrong or the merge is particularly nasty. Do the right away after creating your repo. 
```
git remote add upstream https://github.com/Optimal-Control-16-745/hw3
git pull upstream main --allow-unrelated-histories
```
