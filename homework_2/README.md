# Description
Author: HE KAI LIM
Course: MAE 263F (Flexible Structures and Soft Robots), UCLA Fall 2024

This ipynb notebook is my submission for homework 2.
The script is run from the local python virtual environment (3.10)
The notebook is constructed with *scale* in mind, by incorporating all helper scripts in the folder `helper_functions` and pythonic-ally loading all these functions in the `__init__.py` file, and importing them into the main `.ipynb` notbook with `import helper_functions as helper`. 

Whenever these helper functions are used, they are thus called *like* an object-oriented class, for example, with `helper.computeTangent(q)`. 

This repo folder also contains my report compiled through texlive with a VSCode front end.

# How to Run this Code for the Homework
The code is simply run in its entirety. Where various parameters of refinining the mesh are desired (*dt* *K*), the parameters are manually adjusted.

# Misc. Notes
* Use python black linter and formatter (https://code.visualstudio.com/docs/python/formatting)
* Cupy: I experimented with implementing Cupy to speed up computation, but it is not fully working in this homework, so Numpy is used instead.
