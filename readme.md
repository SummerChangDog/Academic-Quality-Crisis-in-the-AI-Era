# 1.Data Collection Module(Dataset): 

## fetch_index.py
The sample_arxiv_ids function invokes _fetch_ids_for_category_year to retrieve paper IDs from the arXiv platform for two disciplines: cs.AI (Computer Science - Artificial Intelligence) and math (Mathematics). Specifically, it fetches 1000 paper IDs per year for the period 2017 to 2026. After deduplicating and sorting the collected IDs, the function returns a structured result.

## fetch_paper.py
Based on the existing arXiv ids, download the full texts of papers and extract references, then save them into body.txt and ref.json respectively.

## parse_affiliation.py
Identify the affiliation types (academic institution / industry) of the paper authors based on arXiv paper IDs, and save the results to the paper information files.

# 2.Building paper metrics and Calculating the citation hallucination rate

## quality_metrics
The four files in the quality_metrics folder respectively evaluate four textual quality metrics of academic papers (**empirical clarity, distinction between explanation and speculation, linguistic normativity, and mathematical rationality**). The evaluation method involves first writing a prompt to instruct Gemini-2.5-Flash（its url is in **api.py**） to score the paper's quality in each dimension (an integer between 1 and 10, where a higher score indicates poorer quality).

## metadata.py & ref_ai.py
Parse all citations of a paper using **metadata.py**. After that, use **ref_ai.py** to verify the authenticity of references in arXiv papers and calculate the citation hallucination rate for each paper (classified into three levels: L0, L1, L2).

# 3.Collect and organize data

## get_metrics.py
For papers in the math and cs.AI disciplines from a specified year, this script implements a full pipeline that **downloads papers → conducts multi-dimensional quality assessment → verifies reference authenticity → saves results**.
It finally outputs structured evaluation results for each paper (one file per paper), including **4 textual quality metrics + 1 citation hallucination rate + core paper metadata such as ID, discipline, and year**.

Use the following commands to download arxiv papers:
`python pipeline/get_metrics.py`

## create_csv.py
Iterate through all JSON files containing paper evaluation results, extract the core fields, and consolidate the scattered data from these JSON files into a single structured CSV file.

Use the following commands to create csv:
`python pipeline/create_csv.py`

# 4.data visualization

## arxiv_submission.py
Plot the trend of monthly arXiv submission volumes.

Use the following commands to plot:
`python data visualization/arxiv_submission.py`

## quality_trend.py
Plot the annual evolution trend of paper quality metrics for the math and cs.AI disciplines.

Use the following commands to plot:
`python data visualization/quality_trend.py`

## comparative_boxplot.py
Plot a comparative boxplot of the citation hallucination rate distribution for papers in the math and cs.AI fields.

Use the following commands to plot:
`python data visualization/comparative_boxplot.py`

# 5.train models

## test_of_quality_metrics.py
- Plot the correlation coefficient matrix for the four indicators
- Conduct indicator reliability test
- Perform exploratory factor analysis (EFA)

Use the following commands to train:
`python models/test_of_quality_metrics.py`

## profile analysis.py
Compare the "shape differences" in the paper quality dimensions between academia and industry.

Use the following commands to train:
`python models/profile_analysis.py`

## LMM.py
By centering the year variable with 2017 as the baseline, a mixed-effects model incorporating the main effects of domain and year as well as their interaction effect is constructed to quantitatively analyze the differences in citation hallucination rates of papers across different domains and their variation patterns over time.

Use the following commands to train:
`python models/LMM.py`

## critic.py
Preprocess four paper quality indicators (reverse negative indicators to positive, winsorize extreme values, and perform quantile standardization) and then apply the CRITIC method to calculate objective weights for these indicators, finally compute a comprehensive paper quality score.

Use the following commands to train:
`python models/critic.py`

## MESM.py
Implement a multi-level structural equation modeling (MSEM) approach to test the mediating effects of four paper quality indicators on the relationship between citation AI rate (independent variable) and overall paper quality (dependent variable), involving data preprocessing (centering, winsorization, missing value imputation), assumption testing (ICC, normality, multicollinearity), mixed-effects model fitting for path analysis, mediation effect decomposition, and visualization of mediation contribution ratios.

Use the following commands to train:
`python models/MESM.py`