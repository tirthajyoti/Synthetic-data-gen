# Synthetic-data-gen
Various methods for generating synthetic data for data science and ML.

Read my article on Medium **"[Synthetic data generation — a must-have skill for new data scientists](https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae)"**

---
## Notebooks

* [Scikit-learn data generation (regression/classification/clustering) methods](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Scikit-learn-data-generation.ipynb)
* [Random regression and classification problem generation from symbolic expressions (using `SymPy`)](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Symbolic%20regression%20classification%20generator.ipynb)
* [Synthesizing time series](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Synth_Time_series.ipynb)
* [Generating Gaussian mixture model data](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/GMM_generator.ipynb)

## What is synthetic dataset
As the name suggests, quite obviously, a synthetic dataset is a repository of data that is generated programmatically. So, it is not collected by any real-life survey or experiment. Its main purpose, therefore, is **to be flexible and rich enough to help an ML practitioner conduct fascinating experiments with various classification, regression, and clustering algorithms**. Desired properties are,

* It can be numerical, binary, or categorical (ordinal or non-ordinal),
* The number of features and length of the dataset should be arbitrary
* It should preferably be random and the user should be able to choose a wide variety of statistical distribution to base this data upon i.e. the underlying random process can be precisely controlled and tuned,
* If it is used for classification algorithms, then the degree of class separation should be controllable to make the learning problem easy or hard,
* Random noise can be interjected in a controllable manner
* For a regression problem, a complex, non-linear generative process can be used for sourcing the data
