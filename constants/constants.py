LOAN_AMOUNT = 10_000  # Fixed loan amount for profit calculation

# Constants of the datasets (column names, target variable, etc.)

TARGET = "default"  # 0 = no default, 1 = default
SEX = "sex"  # M and F
EMPLOYMENT = "employment"  # employed, unemployed, student, retired
MARRIED = "married"  # 0 = not married, 1 = married
INCOME = "income"  # yearly income in USD
SIGNAL_1 = "signal1"
SIGNAL_2 = "signal2"
SIGNAL_3 = "signal3"

CATEGORICAL = [SEX, EMPLOYMENT, MARRIED]
NUMERICAL = [INCOME, SIGNAL_1, SIGNAL_2, SIGNAL_3]


# Columns for prediction
ID = "id"
DEFAULT_PROBABILITY = "default_probability"
BREAK_EVEN_RATE = "break_even_rate"
LINEAR_RATE = "linear_rate"
QUADRATIC_RATE = "quadratic_rate"


# Columns for market simulation
BORROWER_TYPE = "borrowertype"
BANK_1_RATE = "competing1"
BANK_2_RATE = "own"
BANK_3_RATE = "competing3"
WINNER = "winner"

# Contants for evaluation
PROFIT = "profit"
