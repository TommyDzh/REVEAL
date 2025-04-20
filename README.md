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


We provide the raw CSV data for **GitTablesDB** in `./data/gt-semtab22-dbpedia-all` and for **GitTablesSC** in `./data/gt-semtab22-schema-property-all`.

The **SOTAB-CTA** and **SOTAB-CPA** datasets can be downloaded from the [official SOTAB repository](https://webdatacommons.org/structureddata/sotab/).

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

3. Train evaluate **REVEAL+** model, run:
    ```train
    python run_train_verification.py
    ```

## 🙏 Acknowledgement
This project builds upon ideas and components from the following prior works:
- [Watchog](https://github.com/megagonlabs/watchog)
- [Doduo](https://github.com/megagonlabs/doduo)

We thank the authors for open-sourcing their implementations.