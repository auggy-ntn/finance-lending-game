# HEC Paris: The Lending Game 💰

This repository contains the code for the **Lending Game**, a group project for the *Introduction to Finance for Data Scientists* course. Our team operates as a fintech startup, competing against two other teams to provide €10,000 loans to a large pool of applicants. The primary objective is to maximize total profit by accurately predicting default risk and setting optimal interest rates.

---

## Game Dynamics

### Competition & Borrower Choice
The market consists of our team and two other student teams. Loan applicants are divided into three equal groups, each with a built-in preference for one of the lenders. This preference allows them to accept an offer even if it's up to **2% more expensive** than the lowest available rate.

For example, a "Type 2" applicant, who prefers our team (Lender 2), evaluates offers based on the following adjusted rates and chooses the lowest:
* Our offer: $r(1,i) - 0.02$ 
* Lender 1's offer: $r(2,i)$ 
* Lender 3's offer: $r(3,i)$ 

This dynamic means that while our pricing must be competitive, it doesn't always have to be the absolute lowest to win a loan.

### Profit & Loss
The financial outcomes are straightforward:
* **Loan Repaid**: If a borrower repays, our profit is the interest earned: $Profit = r \times 10,000€$.
* **Loan Default**: If a borrower defaults, we lose the entire principal amount: $Profit = -10,000€$.

---

## The Data

We work with two main datasets.

### `PastLoans.csv` (Training Data)
This historical dataset is used to train our credit scoring model. It contains borrower characteristics and, crucially, includes the private signals from all three lenders (`signal1`, `signal2`, `signal3`), which allows us to build a comprehensive default prediction model.

### `NewApplications_Lender2_RoundX.csv` (New Applicants)
This file contains the 100,000 new loan applications we must price. For these applicants, we only have access to our own team's private signal, not those of our competitors. This information asymmetry is a core strategic challenge of the game.

---

## Core Strategy & Code

Our strategy is based on a two-step, data-driven pricing model.

1.  **Default Probability Prediction**: We first build a model to estimate a continuous probability of default ($PD_i$) for each applicant. This is more effective than a simple binary classification.
2.  **Break-Even Pricing**: With the default probability, we calculate the **break-even interest rate** ($\overline{r}_i$)—the minimum rate required to cover the expected loss from default.The formula is:
    $\overline{r}_i = \frac{PD_i}{1 - PD_i}$
3.  **Final Offer Rate**: Our final offered rate is the break-even rate plus a strategic profit margin. This margin is set to balance profitability against the need to stay competitive and win loan applications.


## Installation & Usage

### Development
To reproduce the development environment, follow these steps:

0. **(Prerequisite)** Have ```uv``` installed. See [the project's website](https://docs.astral.sh/uv/) for more information. In your terminal (MacOS and Linux users), run 
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone the project:
```zsh
git clone https://github.com/auggy-ntn/finance-lending-game
```

2. In the project's workspace run the following command to synchronize your environment with the project's development requirements:
```zsh
uv sync --dev
```
You are all set!

Alternatively, if you don't want to use ```uv```, you can run the following command:
```zsh
pip install -r requirements.txt
```

### Developing with uv

If you work on the project and want to add a package, simply run
```zsh
uv add <package>
``` 
which will update the ```pyproject.toml``` file and the ```uv.lock``` file used by ```uv``` to sync the environment when you run ```uv sync```.

To generate the updated ```requirements.txt``` file, run the following command
```zsh
uv pip freeze > requirements.txt
```

Commit and push these new files to GitHub for others to replicate your environment.