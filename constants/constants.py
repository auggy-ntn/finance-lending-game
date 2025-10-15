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
