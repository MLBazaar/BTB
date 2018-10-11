### Selectors

Selectors apply multiple strategies to decide which models or families of models to
train and test next based on how well thay have been performing in the previous test runs.
This is an application of what is called the Multi-armed Bandit Problem.

`btb.selection` defines Selectors: classes for choosing from a set of discrete options with multi-armed bandits.

The process works by letting know the selector which models have been already tested
and which scores they have obtained, and letting it decide which model to test next.

