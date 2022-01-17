Glossary:

- Run_x is the x implementation of a complete incremental run.
- Level: part of a Run with a fixed number of operators in the spin-partition. Increasing the level means increasing the number of operators.
- Irun: an individual run inside a specific level.


Protocol for incremental runs

1. Throughout a Run modify only the hyperparameter file.

2. Keep a record of different hyperparameter files within a Run in the Level folder within each Run_x of each project.

3. Keep a numbers file in the Run_x folder within the project folder (e.g. in `Run_1' folder of `2d S1 runs' project folder) with the overall current updated data. As the level is increased this file is updated to include more and more CFT data.

4. Break up the run in levels. Different levels means different spin-partitions. Keep record of individual runs (Iruns) at each level in a txt file in project folder/Run_x/records.txt. The protocol for Iruns at a fixed level is the following:

- Update hyperparameter file with new spin-partition and previous CFT data frozen.
- Do 3 Iruns with the added operator. Record the results of the 3 Iruns in records.txt.
- Select the result with the best reward.
- Modify hyperparameter file to unfreeze all the CFT data.
- Starting with the previous best-reward result, do 3 Iruns will all data unfrozen. Record the results in records.txt.
- Select the result with the best reward as the seed for the Iruns of the next level.
- Repeat the above process for the next level starting with the addition of one more operator. Add new operators spin-by-spin across both channels (starting with the lower spins). When the maximum scaling dimension in your spin-partition exceeds the unitarity bound of the next available spin include the addition of operators of that spin as well. 