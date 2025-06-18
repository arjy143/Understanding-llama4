from tokeniser.tokeniser import Tokeniser

# corpus = [
#     "This is the first document.",
#     "This document is the second document.",
#     "And this is the third one.",
#     "Is this the first document?"
# ]

corpus = ["""Other neural network architectures such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) will not be a major focus of this paper. CNNs are more useful in image recognition problems, and RNNs are more useful in cases needing sequential data. This shows that an LSTM approach may be just as useful as an MLP approach for benchmarking certain problem types. 
Give background about optimisation problems. Talk about how many can be solved with maths, but machine learning methods can be more effective in many areas. Explain the background of the different types of problems that can be solved.
Optimisation problems
.  For example, in the case of integer optimisation, the Long Short Term Memory (LSTM) approach to RNNs may be useful. According to this paper [https://arxiv.org/pdf/2207.02937], using an LSTM optimisation framework allowed the researchers to outperform typical MLPs in mixed integer programs by a factor of 9. 
1.	Linear Programming (LP): LP problems seek to maximize or minimize a linear objective function subject to linear constraints. They are typically solved using deterministic mathematical methods like the simplex algorithm or interior-point methods. However, in real-world scenarios with dynamic or noisy data, machine learning methods may offer greater adaptability.
2.	Integer Programming (IP): In IP problems, some or all decision variables are restricted to integer values. These problems are widely used in scheduling, logistics, and resource allocation. While exact methods like branch-and-bound are effective for small instances, larger instances often require heuristics or approximation algorithms, where machine learning can contribute.
3.	Non-Convex Programming: These involve objective functions or constraints that are non-convex, making the search space more complex due to the presence of multiple local optima. Gradient-based methods often struggle with these problems, but neural networks and other machine learning models, which can approximate highly non-linear relationships, have shown promise in exploring such challenging landscapes.
4.	Stochastic Programming: Stochastic optimization deals with uncertainty in problem parameters, often modeled using probability distributions. These are used in financial planning, supply chain management, and energy systems. Traditional approaches like scenario analysis and dynamic programming can become computationally prohibitive as the problem size grows, making data-driven methods attractive alternatives.
Machine learning methods, including neural networks, have emerged as powerful tools for optimization. They excel in high-dimensional problems and scenarios where traditional solvers struggle to generalize or adapt to variations. By learning patterns from data, neural networks can approximate solutions to problems that are otherwise computationally intensive. For instance, reinforcement learning has been applied to combinatorial optimization problems, while neural networks have been used for surrogate modeling in non-convex problems.
This project explores whether Kolmogorov-Arnold Networks (KANs) can bridge the gap between traditional optimization methods and machine learning by leveraging their unique structure and capabilities. By evaluating KANs across a range of optimization problem types, this research aims to uncover their potential as a general-purpose optimization tool.
Analysis section
Analysis of MLPs in solving optimisation problems, and potential advantages for KANs.
Multi-layer perceptrons (MLPs) have been widely studied and applied to optimization problems due to their universal function approximation capabilities. In particular, MLPs can model complex, non-linear relationships and are highly adaptable to a variety of problem domains. However, their performance is often contingent on careful tuning of hyperparameters, large training datasets, and sufficient computational resources.
Analysis of MLPs
Will possibly need to write sections for each type of problem
MLPs have been successfully used in optimization tasks such as surrogate modeling for non-convex problems, heuristic-based optimization, and learning-based methods for integer and combinatorial optimization. For example, reinforcement learning algorithms powered by MLPs have demonstrated efficacy in traveling salesman problems and job-shop scheduling tasks. These models leverage their ability to approximate cost or reward functions, enabling faster exploration of solution spaces than traditional methods.
However, MLPs are not without drawbacks. Their reliance on gradient-based optimization methods, such as stochastic gradient descent (SGD), can make them susceptible to getting trapped in local minima, particularly in non-convex problems. Furthermore, MLPs typically require significant retraining or fine-tuning when applied to new problem instances, limiting their flexibility. Another challenge lies in their "black-box" nature, which reduces interpretability and can make diagnosing failures or improving performance difficult.

is a widely studied optimization domain with well-established solution techniques. The Simplex method, introduced by Dantzig (1947), remains one of the most effective approaches for solving LP problems due to its deterministic nature and ability to traverse feasible solutions efficiently. It is particularly robust for small- to medium-scale problems and provides exact solutions. However, Simplex suffers from exponential worst-case time complexity in theory, even though practical performance is generally satisfactory for most real-world cases (Klee & Minty, 1972).
The Interior Point method, developed later, offers polynomial time complexity and is more effective for solving large-scale LP problems (Karmarkar, 1984). It approaches the solution by traversing the interior of the feasible region, making it competitive with Simplex in many cases.
Summarise
To summarise, many pieces of literature show significant advancements in the use of traditional neural networks for solving optimization problems, and traditional MLPs demonstrate robust performance in both linear and non-linear contexts. However, they have limitations in terms of scalability and efficiency, highlighting the need for different approaches. Kolmogorov-Arnold networks (KANs), with their unique B-spline activation functions, present a promising solution, offering improved representational power and computational efficiency. While initial studies, such as those by Xiaoming et al. (2023), suggest the theoretical advantages of KANs, there are few empirical evaluations across diverse optimization domains. Moreover, there is limited to no research directly comparing KANs with traditional neural network architectures in practical scenarios, particularly in complex problem types like integer, non-convex, and stochastic programming. This project seeks to address these gaps by conducting a systematic investigation of KANs’ performance across a range of optimization problems, providing a comprehensive analysis of their capabilities and potential applications.
"""]
from datasets import load_dataset


# 1. Load a dataset with lots of text
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(dataset.shape)
# 2. Combine all text samples into one big list
all_texts = dataset["text"]
tokeniser = Tokeniser(merges=10000)
tokeniser.train(all_texts)

# text = "is this document the third one?"
# tokens = tokeniser.encode(text)

# tokeniser.decode(tokens)