# automatic-code-review
* The goal of the project is to partially automate the code review process.
* This means that comments are automatically added to the code.
* The idea is to let RL agent interact with the environment represented by the abstract syntax tree of the code.
* The agent can perform actions that represent labels or classes (comments) on the code.
* The environment then gives the agent a reward based on the chosen action.
* 
The main idea is to use RL algorithms on the abstract sytnax tree generated on python code
Each code block is an episode.
The agent should find out if the code is ok or not.

### Problems:
1. Most of the inputs are strings, how to encode them
2. How to represent the states.
3. how to enable the agent to memorize previous states
