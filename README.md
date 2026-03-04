# Retrieve-and-Verify: A Table Context Selection Framework for Accurate Column Annotations

This repository contains the official codebase for our paper:  
**"Retrieve-and-Verify: A Table Context Selection Framework for Accurate Column Annotations."**

In this work, we propose a novel retrieve-and-verify context selection framework for accurate column annotation, covering both **Column Type Annotation (CTA)** and **Column Property Annotation (CPA)** tasks (also referred to as Column Relation Annotation). The framework consists of two methods: **REVEAL** and **REVEAL+**.

- **REVEAL** includes a retrieval stage that selects a compact and informative subset of column context for a given target column by balancing semantic relevance and diversity. It also introduces context-aware encoding techniques to distinguish target and context columns, enabling effective contextualized column representations.

- **REVEAL+** extends REVEAL by introducing a lightweight verification model that refines the selected context by directly estimating its quality for a specific annotation task. It formulates column context verification as a supervised classification problem, and incorporates a top-down inference strategy to efficiently reduce the search space for high-quality context subsets from exponential to quadratic complexity.

---

## 🚀 Requirements

To install the required dependencies:

```bash
pip install -r requirements.txt
```

##  📊 Datasets
The following table summarizes the datasets used in our experiments:

| Benchmark     | # Tables | # Types | Total # Cols | # Labeled Cols | Min/Max/Avg Cols per Table |
|---------------|----------|---------|---------------|----------------|-----------------------------|
| GitTablesDB        | 3,737    | 101     | 45,304        | 5,433          | 1 / 193 / 12.1              |
| GitTablesSC        | 2,853    | 53      | 34,148        | 3,863          | 1 / 150 / 12.0              |
| SOTAB-CTA     | 24,275   | 91      | 195,543       | 64,884         | 3 / 30 / 8.1                |
| SOTAB-CPA     | 20,686   | 176     | 196,831       | 74,216         | 3 / 31 / 9.5                |
| WikiTable-CTA      | 406,706  | 255     | 2,393,027     | 654,670        | 1 / 99 / 5.9                |
| WikiTable-CPA     | 55,970   | 121     | 306,265       | 62,954         | 2 / 38 / 5.5                |


We make our processed data for all 6 datasets publicly available on [Huggingface Repo](https://huggingface.co/datasets/Tommy-DING/REVEAL).

The original **SOTAB-CTA** and **SOTAB-CPA** datasets can be downloaded from the [official SOTAB repository](https://webdatacommons.org/structureddata/sotab/).

> **Note:** The dataset names used in our paper and the corresponding task identifiers in the codebase are listed below:

| Paper Name        | Codebase Task Name                     |
|-------------------|-----------------------------------------|
| GitTablesDB       | `gt-semtab22-dbpedia-all`              |
| GitTablesSC       | `gt-semtab22-schema-property-all`      |
| SOTAB-CTA         | `sotab`                                |
| SOTAB-CPA         | `sotab-re`                             |
| WikiTables-CTA    | `turl`                                 |
| WikiTables-CPA    | `turl-re`                              |

### 🔥 Training
1. Train and evaluate **REVEAL** model, run:
    ```train
    python run_train_reveal.py
    ```
2. Construct data for verification model
    ```train
    python construct_verification_data.py --task [dataset_name] --best_dict_path [dir_path]
    ```
    - `dataset_name`: Name of the dataset/task (e.g., `gt-semtab22-dbpedia-all`, `sotab`)
    - `dir_path`: Path to the trained REVEAL model checkpoint

3. Train and evaluate **REVEAL+** model, run:
    ```train
    python run_train_verification.py
    ```
## 📌 Citation
```
@article{10.1145/3769823,
  author = {Ding, Zhihao and Sun, Yongkang and Shi, Jieming},
  title = {Retrieve-and-Verify: A Table Context Selection Framework for Accurate Column Annotations},
  year = {2025},
  issue_date = {December 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {3},
  number = {6},
  url = {https://doi.org/10.1145/3769823},
  doi = {10.1145/3769823},
  abstract = {Tables are a prevalent format for structured data, yet their metadata, such as semantic types and column relationships, is often incomplete or ambiguous. Column annotation tasks, including Column Type Annotation (CTA) and Column Property Annotation (CPA), address this by leveraging table context, which are critical for data management. Existing methods typically serialize all columns in a table into pretrained language models to incorporate context, but this coarse-grained approach often degrades performance in wide tables with many irrelevant or misleading columns. To address this, we propose a novel retrieve-and-verify context selection framework for accurate column annotation, introducing two methods: REVEAL and REVEAL+. In REVEAL, we design an efficient unsupervised retrieval technique to select compact, informative column contexts by balancing semantic relevance and diversity, and develop context-aware encoding techniques with role embeddings and target-context pair training to effectively differentiate target and context columns. To further improve performance, in REVEAL+, we design a verification model that refines the selected context by directly estimating its quality for specific annotation tasks. To achieve this, we formulate a novel column context verification problem as a classification task and then develop the verification model. Moreover, in REVEAL+, we develop a top-down verification inference technique to ensure efficiency by reducing the search space for high-quality context subsets from exponential to quadratic. Extensive experiments on six benchmark datasets demonstrate that our methods consistently outperform state-of-the-art baselines.},
  journal = {Proc. ACM Manag. Data},
  month = dec,
  articleno = {358},
  numpages = {27},
  keywords = {column annotation, context selection, embeddings, table understanding}
}
```
## 🙏 Acknowledgement
We acknowledge the open-sourced implementations of [Watchog](https://github.com/megagonlabs/watchog) and [Doduo](https://github.com/megagonlabs/doduo), which provide basic componenents partially used in our implementation.
