# Credit Card Approval Prediction

A dataset and toolkit for predicting credit card approval using machine learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Objectives](#objectives)
- [Features & Challenges](#features--challenges)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook / App](#running-the-notebook--app)
- [Project Structure](#project-structure)
- [Suggested Enhancements](#suggested-enhancements)
- [License & Acknowledgements](#license--acknowledgements)

---

## Project Overview

This repository includes:

- **`application_record.csv`**: Contains client demographics and employment history.
- **`credit_record.csv`**: Monthly payment status over time, indicating overdue patterns and repayment reliability.
- **Use Case**: Build ML models to assess the likelihood of credit card approval based on historical behavior.

---

## Dataset Details

### 1. `application_record.csv`

Merged via a shared `ID` with `credit_record.csv`. Key columns include:

- `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
- `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`
- `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `DAYS_BIRTH`, `DAYS_EMPLOYED`
- `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`
- `OCCUPATION_TYPE`, `CNT_FAM_MEMBERS`

### 2. `credit_record.csv`

Tracks monthly repayment status, with values:

- `0`: 1–29 days past due
- `1`: 30–59 days overdue
- `2`: 60–89 days past due
- `3`: 90–119 days past due
- `4`: 120–149 days overdue
- `5`: Overdue/write-off >150 days
- `C`: Paid off this month
- `X`: No loan that month

---

## Objectives

- **Merge** both datasets using `ID` for comprehensive client profiles.
- **Engineer features** such as default frequency, trend of late payments, or flags for ‘good’ vs. ‘bad’ clients.
- **Address class imbalance** and define an appropriate labeling strategy—“good” vs. “bad”—e.g., via vintage analysis or threshold-based criteria.

---

## Features & Challenges

- **Class imbalance**: A typical challenge in credit behavior datasets.
- **Label ambiguity**: No explicit “good/bad” label; needs creative derivation such as past-due thresholds or recency analysis.
- **Temporal nature**: Leverage monthly patterns to derive meaningful features.

---

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook (or your preferred editor)
- (Optional) Virtual environment tool: `venv` or `conda`

### Installation

```bash
git clone https://github.com/jayuan101/Credit-Card-Approval-Prediction.git
cd Credit-Card-Approval-Prediction
pip install -r requiredment.txt
