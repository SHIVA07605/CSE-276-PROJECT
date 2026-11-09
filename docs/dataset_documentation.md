# Dataset Documentation & Provenance

## 1. Problem Context
Public Sector Enterprises (e.g., power utilities, railways, oil & gas PSUs, municipal corporations) maintain disparate procurement catalogues. Materials that are identical in technical specification have different primary keys and free-text descriptions.

## 2. Core Schema
Every material record ingested into the pipeline must conform to or be mapped to the following standard schema:

| Field Name | Type | Description | Mandatory |
| :--- | :--- | :--- | :--- |
| `enterprise_name` | String | Source organisation or enterprise identifier (e.g. `Enterprise_A`) | Yes |
| `material_code` | String | Internal primary key / stock code in source enterprise | Yes |
| `material_description`| String | Unstructured text description of the material item | Yes |
| `category` | String | Broad taxonomy/commodity classification (if available) | Optional |
| `uom` | String | Unit of measurement (e.g. `EA`, `SET`, `KGS`, `MTR`) | Optional |
| `manufacturer` | String | Brand or original equipment manufacturer (OEM) | Optional |
| `specifications` | String | Technical standards (e.g. `IS 1367`, `DIN 933`, `AISI 316`) | Optional |

## 3. Labelled Pairs Schema (Ground Truth)
Used for evaluating pairwise Precision, Recall, and F1:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `record_id_1` | String | Combined ID (`enterprise_name + '_' + material_code`) for item 1 |
| `record_id_2` | String | Combined ID for item 2 |
| `label` | Integer | Binary ground truth (1 = functionally equivalent, 0 = non-equivalent) |
| `canonical_cluster_id`| String | Ground-truth cluster ID grouping all equivalent items |
| `annotation_source` | String | Manual verification / domain expert audit / benchmark reference |

## 4. Integrity and Provenance Policy
- No synthetic datasets or fabricated metrics are allowed.
- Any benchmark datasets (e.g., Abt-Buy, WDC, GeM public samples) used for transfer testing or simulation must be clearly documented with their license and origin.
- A general e-commerce dataset must not be falsely represented as PSU procurement data.
