# Retrieve-and-Verify: A Table Context Selection Framework for Accurate Column Annotations

This repository contains the codebase of our paper "Retrieve-and-Verify: A Table Context Selection Framework for Accurate Column Annotations".

In this work, we present a novel retrieve-and-verify context selection framework for accurate column annotation, including Column Type Annotation (CTA) task and Column Property Annotation (CPA) or Column Relation Annotation. The frameworks consists of two methods: **REVEAL** and **REVEAL+**. 

REVEAL consists of a retrieval stage for selecting compact, informative column context for a target  by balancing semantic relevance and diversity, and develop
 context-aware encoding techniques to differentiate target and context columns for learning contextualized column representation.   

 REVEAL+ extends REVEAL by introducing a verification model that refines the selected context by
 directly estimating its quality for specific annotation tasks, through
 a novel formulation of column context verification as a supervised
 classification task. To ensure efficiency,  REVEAL+ incorporates a top-down
 inference method that reduces the search space for high-quality
 context subsets from exponential to quadratic complexity. 


 ## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

| Benchmark     | # Tables | # Types | Total # Cols | # Labeled Cols | Min/Max/Avg Cols per Table |
|---------------|----------|---------|---------------|----------------|-----------------------------|
| \gitdb        | 3,737    | 101     | 45,304        | 5,433          | 1 / 193 / 12.1              |
| \gitsc        | 2,853    | 53      | 34,148        | 3,863          | 1 / 150 / 12.0              |
| \sotabcta     | 24,275   | 91      | 195,543       | 64,884         | 3 / 30 / 8.1                |
| \sotabcpa     | 20,686   | 176     | 196,831       | 74,216         | 3 / 31 / 9.5                |
| \wikicta      | 406,706  | 255     | 2,393,027     | 654,670        | 1 / 99 / 5.9                |
| \wikicpa      | 55,970   | 121     | 306,265       | 62,954         | 2 / 38 / 5.5                |

The data for SOTAB-CTA and SOTAB-CPA can be downloaded [here](https://webdatacommons.org/structureddata/sotab/).



## Acknowledgement

In code implmentation, we refer to [Watchog](https://github.com/megagonlabs/watchog) and [Doduo](https://github.com/megagonlabs/doduo). 