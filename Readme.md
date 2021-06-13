<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo_name" >
    <!-- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Aco_shortpath.svg/330px-Aco_shortpath.svg.png" alt="Logo" width="210"> -->
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Hypercubematrix_binary.svg/180px-Hypercubematrix_binary.svg.png" style="border-radius:50%;background-color:#8895b97d" alt="Logo" width="210">
  </a>

  <h3 align="center">Logical Feature construction for Naive Bayes</h3>

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
<strong>RankerLogicalFeatureConstructor</strong> - for the Hybrid Ranker-Wrapper.
</li>
<li>
<strong>ACFCS</strong> -for the Ant Colony-based algorith. 
</li>
<li>
<strong>GeneticAlgorithm</strong> -for the Genetic Programming.
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

from tfg.genetic_programming import GeneticProgrammingV3
from tfg.naive_bayes import NaiveBayes


# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    shuffle=True, 
                                                    stratify=y)


# Train model
clf = GeneticProgrammingV3(seed=0, individuals=20, generations=30)
clf.fit(X_train, y_train)

# Obtain accuracy
accuracy_score = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy_score}")
```

<!--
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.
-->
<!--_For more examples, please refer to the [Documentation](https://example.com)_ -->

<!-- LICENSE
## License

Distributed under the MIT License. See `LICENSE` for more information.
-->

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
This work as been developed under the tutorship of Jose Antonio Gámez and Juan Ángel Aledo at Universidad de Castilla La-Mancha.






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username -->
