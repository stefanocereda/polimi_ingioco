# polimi_ingioco
This repository contains the dataset (and an example solution) for the PoliMi event ``Mettiti in gioco: scopri come risolvere casi di Machine Learning'': https://www.careerservice.polimi.it/it-IT/Meetings/Home/Index/?eventId=23667

## Problem definition
A modern IT system is composed by many IT layers, each layer has tunable configuration parameters affecting performance.
We need to find the optimal configuration, but the search space is huge.

A possible solution consists in running a random search on a separate test environment.
Once we find a good configuration we move it to the production environment.
However, changing the configuration is risky, and so we need to minimize the number of configuration parameters to modify.

Here you have a sample dataset, collected on MongoDB tuning MongoDB and Linux parameters.
The first column is the experiment id, then we have the throughput (which is our target performance metric, to be maximized) and then all the explored parameters.
The first line contains the baseline, which is the vendor default configuration.
Your goal is to suggest a subset of parameters to modify, keeping as many parameters as possible to their default setting, while still improving performance.

If you want, you can submit your solution with a pull request.
