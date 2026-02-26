"""
Configuration constants and default parameters for the Dataset Analysis MCP Server.

This module centralizes all configuration values to avoid duplication across the codebase
and provide a single source of truth for server settings.
"""

import os

# ============================================================================
# Directory Configuration
# ============================================================================

# Project root directory (where this config.py file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory for storing input/output datasets
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================================
# Server Configuration
# ============================================================================

# MCP Server name
SERVER_NAME = "Dataset Analysis MCP"

# ============================================================================
# Data Quality Detection Parameters
# ============================================================================

# Missing values severity thresholds (percentage)
MISSING_VALUES_THRESHOLDS = {
    "low": 5.0,      # < 5% missing
    "medium": 20.0,  # 5-20% missing
    # > 20% is considered high
}

# Outlier detection severity thresholds (percentage)
OUTLIER_THRESHOLDS = {
    "low": 1.0,      # < 1% outliers
    "medium": 5.0,   # 1-5% outliers
    # > 5% is considered high
}

# High cardinality thresholds
HIGH_CARDINALITY_RATIO = {
    "low": 0.5,      # > 50% unique values
    "medium": 0.8,   # > 80% unique values
    "high": 0.95,    # > 95% unique values
}

HIGH_CARDINALITY_ABSOLUTE = {
    "low": 50,       # > 50 unique categories
    "medium": 100,   # > 100 unique categories
    "high": 200,     # > 200 unique categories
}

# Duplicate rows severity thresholds (count)
DUPLICATE_ROWS_THRESHOLDS = {
    "low": 10,       # < 10 duplicates
    "medium": 100,   # 10-100 duplicates
    # > 100 is considered high
}

# ============================================================================
# Outlier Removal Default Parameters
# ============================================================================

# Default Z-score threshold (standard deviations from mean)
DEFAULT_ZSCORE_THRESHOLD = 3.0

# Default IQR multiplier for fence calculation
DEFAULT_IQR_MULTIPLIER = 1.5

# Default Modified Z-score threshold (3.5 is standard recommendation)
DEFAULT_MODIFIED_ZSCORE_THRESHOLD = 3.5

# Default Random State for reproducibility
DEFAULT_RANDOM_STATE = 42


# ============================================================================
# Statistical Analysis Parameters
# ============================================================================

# Minimum sample size for outlier detection
MIN_OUTLIER_DETECTION_SAMPLES = 3

# Minimum sample size for statistical tests
MIN_SAMPLE_SIZE_STATS = 30

# Distribution thresholds for adaptive method selection
SKEWNESS_THRESHOLD = 1.0     # Absolute skewness >= 1.0 is highly skewed
KURTOSIS_THRESHOLD = 3.0     # Absolute kurtosis >= 3.0 indicates heavy tails

# Normality thresholds (for choosing between Z-score and IQR)
NEAR_NORMAL_SKEWNESS = 0.5   # Absolute skewness < 0.5 is nearly normal
NEAR_NORMAL_KURTOSIS = 1.0   # Absolute kurtosis < 1.0 is nearly normal

# Maximum cardinality for categorical analysis (to avoid ID columns)
MAX_CATEGORICAL_CARDINALITY = 20

# Minimum unique categories for correlation analysis
MIN_UNIQUE_CATEGORIES = 2

# Rare category threshold (percentage)
RARE_CATEGORY_THRESHOLD = 0.05  # Categories appearing in < 5% of rows

# ============================================================================
# Data Export Configuration
# ============================================================================

# Supported export formats for datasets
SUPPORTED_DATASET_FORMATS = ["csv", "json", "parquet"]

# Supported export formats for pipeline configurations
SUPPORTED_PIPELINE_FORMATS = ["json", "yaml"]

# ============================================================================
# Correlation Analysis Parameters
# ============================================================================

# Default correlation method for numerical features
DEFAULT_CORRELATION_METHOD = "pearson"

# Supported correlation methods
SUPPORTED_CORRELATION_METHODS = ["pearson", "spearman", "kendall"]

# ============================================================================
# Categorical Encoding Parameters
# ============================================================================

# One-hot encoding cardinality thresholds
ONE_HOT_MAX_CARDINALITY = 20          # Warn above this
ONE_HOT_BLOCK_CARDINALITY = 100       # Block above this (too many columns)

# Hashing default number of components
DEFAULT_HASH_N_COMPONENTS = 8

# Target encoding default smoothing factor
DEFAULT_TARGET_SMOOTHING = 1.0

# Memory risk thresholds (based on new columns created)
MEMORY_RISK_THRESHOLDS = {
    "low": 10,
    "medium": 50,
    # > 50 is high
}

# ============================================================================
# Categorical Normalization Parameters
# ============================================================================

# Fuzzy clustering: minimum rapidfuzz similarity score (0–100) to merge strings
FUZZY_SCORE_THRESHOLD = 85

# Fuzzy clustering: minimum cluster size to apply replacement
FUZZY_MIN_GROUP_SIZE = 1

# Fuzzy clustering: max unique values before refusing pairwise comparison
# Prevents O(n²) explosion on high-cardinality columns
FUZZY_MAX_COMPARISONS = 500

# Flashtext: whether keyword matching is case-sensitive
HARMONIZE_CASE_SENSITIVE = False

# ============================================================================
# Dataset Versioning Configuration
# ============================================================================

# Maximum number of version snapshots to keep in memory (LRU eviction)
# Version 0 (original load) is auto-pinned and never evicted
MAX_DATASET_VERSIONS = 10
