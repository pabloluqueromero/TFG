<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo_name" >
    <!-- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Aco_shortpath.svg/330px-Aco_shortpath.svg.png" alt="Logo" width="210"> -->
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Hypercubematrix_binary.svg/180px-Hypercubematrix_binary.svg.png" style="border-radius:50%;background-color:#8895b97d" alt="Logo" width="210">
  </a>

  <h3 align="center">Logical Feature Construction for the Naive Bayes classifier</h3>

  <p align="center">
    Welcome to my Bachelor's Final degree project. The aim is to explore different techniques for feature construction and selection using the logical operators : XOR, AND and OR to improve the performance of the Naive Bayes classifier.
    <br />
    <a href="https://github.com/pabloluqueromero/TFG"><strong>Explore the code »</strong></a>
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Welcome to my Bachelor's Final degree project. The aim is to explore different techniques for feature construction and selection using the logical operators : XOR, AND and OR to improve the performance of the Naive Bayes classifier. The three proposed algorithms correspond to the classes: 
<ul>
<li>
<strong>RankerLogicalFeatureConstructor</strong> for the Hybrid Ranker-Wrapper.
</li>
<li>
<strong>ACFCS</strong> for the Ant Colony Optimization algorithm. 
</li>
<li>
<strong>GeneticAlgorithm</strong> for the Genetic Programming.
</li>
</ul>

### Built With

* [Python](https://www.python.org/)
* [Sklearn](https://scikit-learn.org/)
* [Numpy](https://www.numpy.org/)
* [Numba](https://numba.pydata.org/)



<!-- GETTING STARTED -->
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/pabloluqueromero/TFG.git
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage
<p>The algorithms are compatible with sklearn, they can be used like any other classifier from the library.
</p>

 ```python
# Load imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


from tfg.optimization.genetic_programming import GeneticProgrammingRankMutation
from tfg.pazzani import PazzaniWrapper


# Load data
X, y = load_iris(return_X_y=True)

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=y)


clf_genetic = GeneticProgrammingRankMutation(seed=0, individuals=10, generations=10)
clf_pazzani = PazzaniWrapper(strategy="FSSJ")

# Train models
clf_genetic.fit(X_train, y_train)
clf_pazzani.fit(X_train, y_train)

# Obtain accuracy
accuracy_genetic = clf_genetic.score(X_test, y_test)
accuracy_pazzani = clf_pazzani.score(X_test, y_test)
print(f"Genetic Programming Accuracy: {accuracy_genetic}")
print(f"Pazzani Wrapper Accuracy: {accuracy_pazzani}")
```

## Acknowledgements
This work has been developed under the tutorship of José Antonio Gámez and Juan Ángel Aledo Sánchez at Universidad de Castilla La-Mancha.
