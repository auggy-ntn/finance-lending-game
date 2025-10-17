# Paths

from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

PAST_LOANS_PATH = DATA_DIR / "PastLoans.csv"
NEW_LOANS_PATH = DATA_DIR / "NewApplications_Lender2_Round1.csv"

METRICS_PATH = OUTPUT_DIR / "metrics.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

RESULTS_PATH = DATA_DIR / "Round1_Diagnostic_21.csv"
NEW_LOANS_PATH_2 = DATA_DIR / "NewApplications_Lender2_Round2.csv"
