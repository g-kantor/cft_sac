# Welcome to the CFT SAC algorithm.
---

## Getting started

In this readme file we will describe how to quickly get started with using the
code. First things first, you will need to download a working version of the
code which can be done [here](https://github.com/pake-jeralta/cft_sac/releases).

If you want to get the very very latest version (might be under development/not
too stable) you can always clone the repository.

## Running the Code

Running the code is super simple. All you have to do is run in the terminal:

`python run.py`

You might need to change "python" to "python3" depending on you installation of
Python.

Note that the result of the code is written into the *current_result.txt* file.
Every single time the code is run, the top line (the one relating to the reward)
must be changed to 0.

## Changing the Code

All the parameters of the the code are stored in the *hyperparameters.py* file.
It is also possible to change the type of theories one is looking at by
modifying the conformal blocks in the *blocks.py* file.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib
