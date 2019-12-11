# Synthetic-data-gen
Various methods for generating synthetic data for data science and ML.

Read my article on Medium **"[Synthetic data generation — a must-have skill for new data scientists](https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae)"**

---
## Notebooks

* [Scikit-learn data generation (regression/classification/clustering) methods](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Scikit-learn-data-generation.ipynb)
* [Random regression and classification problem generation from symbolic expressions (using `SymPy`)](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Symbolic%20regression%20classification%20generator.ipynb)
* [Synthesizing time series](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Synth_Time_series.ipynb)
* [Generating Gaussian mixture model data](https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/GMM_generator.ipynb)

## Why do you need the skill of synthetic data generation?

Imagine you are tinkering with a cool machine learning algorithm like SVM or a deep neural net. What kind of dataset you should practice them on? If you are learning from scratch, the advice is to start with simple, small-scale datasets which you can plot in two dimensions to understand the patterns visually and see for yourself the working of the ML algorithm in an intuitive fashion. For example, [here is an excellent article](https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/) on various datasets you can try at various level of learning.

This is a great start. But it is not all.

Sure, you can go up a level and find yourself a real-life large dataset to practice the algorithm on. But that is still a fixed dataset, with a fixed number of samples, a fixed pattern, and a fixed degree of class separation between positive and negative samples (if we assume it to be a classification problem). Are you learning all the intricacies of the algorithm in terms of
- sample complexity,
- computational efficiency,
- ability to handle class imbalance,
- robustness of the metrics in the face of varying degree of class separation
- bias-variance trade-off as a function of data complexity

Probably not. **Perhaps, no single dataset can lend all these deep insights for a given ML algorithm**. But, these are extremely important insights to master for you to become a true expert practitioner of machine learning. So, you will need an **extremely rich and sufficiently large dataset, which is amenable enough for all these experimentation**.

So, what can you do in this situation? Scour the internet for more datasets and just hope that some of them will bring out the limitations and challenges, associated with a particular algorithm, and help you learn?

Yes, it is a possible approach but may not be the most viable or optimal one in terms of time and effort. Good datasets may not be clean or easily obtainable. You may spend much more time looking for, extracting, and wrangling with a suitable dataset than putting that effort to understand the ML algorithm.

Make no mistake. **The experience of searching for a real life dataset, extracting it, running exploratory data analysis, and wrangling with it to make it suitably prepared for a machine learning based modeling is invaluable**. I know because I wrote a book about it :-)

But that can be taught and practiced separately. In many situations, however, **you may just want to have access to a flexible dataset (or several of them) to ‘teach’ you the ML algorithm in all its gory details**.

Surprisingly enough, in many cases, such teaching can be done with **synthetic datasets**.

## What is a synthetic dataset?
As the name suggests, quite obviously, a synthetic dataset is a repository of data that is generated programmatically. So, it is not collected by any real-life survey or experiment. Its main purpose, therefore, is **to be flexible and rich enough to help an ML practitioner conduct fascinating experiments with various classification, regression, and clustering algorithms**. Desired properties are,

* It can be numerical, binary, or categorical (ordinal or non-ordinal),
* The number of features and length of the dataset should be arbitrary
* It should preferably be random and the user should be able to choose a wide variety of statistical distribution to base this data upon i.e. the underlying random process can be precisely controlled and tuned,
* If it is used for classification algorithms, then the degree of class separation should be controllable to make the learning problem easy or hard,
* Random noise can be interjected in a controllable manner
* For a regression problem, a complex, non-linear generative process can be used for sourcing the data
