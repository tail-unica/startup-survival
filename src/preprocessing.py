import re

import pandas as pd
import polars as pl

#: The panel columns the models read, in the order the datasets carry them.
#: ``Target`` is built by :func:`build_windowed_dataset` or
#: :func:`build_full_history_dataset`; every other name has to exist in the panel
#: that ``src/panel/pipeline.py`` produces, which is what
#: ``tests/test_integration.py`` checks.
FEATURE_COLUMNS: list[str] = [
    "YearFounded",
    "Age",
    "N_Deal",
    "TotalRaised",
    "UndisclosedAmountShare",
    "Percent_Females",
    "Is_Eco",
    "Is_Eng",
    "Is_NS",
    "Is_Hum",
    "Is_SS",
    "Is_Med",
    "Is_Law",
    "Is_IT",
    "Institute",
    "WorkExp_Idx_Mean",
    "Total_Founders",
    "Is_Debt",
    "Is_SpinOff",
    "Is_CrowdFunding",
    "MeanMedianRoundAmount_cum",
    "Is_Accelerator",
    "has_Corporate",
    "has_VentureCapital",
    "has_PublicInvestor",
    "has_Angel_Lead",
    "has_Corporate_Lead",
    "has_VentureCapital_Lead",
    "has_Accelerator_Lead",
    "has_PrivateEquity_Lead",
    "has_PublicInvestor_Lead",
    "HQCountry",
    "PrimaryIndustrySector",
    "SimilarityScoreMean",
    "N_Competitors",
    "N_Similar",
    "Same_Country",
    "Highest_Degree_CEO",
    "Gender_CEO",
    "MeanTotalInvestments_cum",
    "WorkExperienceIndex_CEO",
    "Is_Angel",
    "Total_People",
    "Is_Grant",
    "has_PrivateEquity",
    "TotalInvestors",
    "Highest_Degree_Mean",
    "Avg_Earliest_Year",
]

#: Columns :func:`build_windowed_dataset` needs to place a firm in time and to read
#: its target. They belong to the panel too, but they never reach the models.
TARGET_COLUMNS: list[str] = [
    "CompanyID",
    "Age",
    "YearFounded",
    "GrowthStageGroup",
    "GrowthNextStageGroup",
    "TimeNextStageGroup",
]


def process_university_list(path):
    """
    This function takes the path of the raw QS world university ranking file, and returns a
    dataframe with the top 50 universities,
    their overall score and the year of the ranking. The function also cleans the university names
    by removing common words and punctuation,
    and extracts the acronym if present.

    :param path: Path of the raw QS world ranking file
    """
    df_uni = pd.read_csv(path)

    df_uni.columns = [col.strip().replace(" ", "_") for col in df_uni.columns]

    df_uni["Overall_Score"] = pd.to_numeric(df_uni["Overall_Score"], errors="coerce")
    df_uni_filtered = df_uni.dropna(subset=["Overall_Score"])

    # Sort the dataframe by University and Year in descending order
    df_uni_sorted = df_uni_filtered.sort_values(by=["University", "Year"], ascending=[True, False])

    # Keep only the first occurrence of each university
    result_df_uni = df_uni_sorted.drop_duplicates(subset=["University"], keep="first")

    result_df_uni = result_df_uni.sort_values(by="Overall_Score", ascending=False)

    final_result_uni = result_df_uni[["University", "Overall_Score", "Year"]]

    top_50_universities = final_result_uni.head(50).copy()

    top_50_universities["Cleaned_University"] = (
        top_50_universities["University"]
        .str.lower()
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace("university", "", regex=False)
        .str.replace("of", "", regex=False)
        .str.replace("the", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # Extract the acronym from the university name if present (text between parentheses)
    top_50_universities["acronym"] = (
        top_50_universities["University"].str.lower().str.extract(r"\((.*?)\)")
    )

    return top_50_universities


def is_in_top50_institutes(university, top_50_universities):
    """
    This function takes a university name and the dataframe of the top 50 universities,
    and returns a boolean flag indicating whether the university is in the top 50 universities.

    :param university: University name to check
    :param top_50_universities: Dataframe of the top 50 universities with cleaned names and acronyms
    """

    top50names = top_50_universities["Cleaned_University"].dropna().unique()
    top50acronyms = top_50_universities["acronym"].dropna().unique()

    match = re.search(r"\((.*?)\)", str(university).lower())
    if match:
        acrP = match.group(1)
    else:
        acrP = None

    uniP = str(university).lower()
    uniP = re.sub(r"\(.*?\)", "", uniP)  # remove text between parentheses
    uniP = (
        uniP.replace("university", "")
        .replace("of", "")
        .replace("the", "")
        .replace(",", "")
        .replace("-", "")
    )

    uniP = re.sub(r"\s+", " ", uniP)  # replace multiple spaces with a single space
    uniP = uniP.strip()  # remove leading and trailing spaces

    if not isinstance(uniP, str):
        uniP = ""
    if not isinstance(acrP, str):
        acrP = ""

    if acrP != "" and acrP in top50acronyms:
        return True

    if uniP != "":
        if uniP in top50names:
            return True

        for acronym in top50acronyms:
            if not isinstance(acronym, str):
                acronym = ""
            if acronym != "" and acronym in uniP:
                return True
        for name in top50names:
            if name in uniP or are_strings_similar(name, uniP):
                return True
    return False


def are_strings_similar(string1, string2, threshold=0.5):
    """
    This function takes two strings and a similarity threshold,
    and returns a boolean flag indicating whether the two strings are similar based on the
    similarity of their words.

    :param string1: First string to compare
    :param string2: Second string to compare
    :param threshold: Similarity threshold (default is 0.5, meaning that the strings are considered
        similar if at least 50% of their words are common)
    """

    similarity = calculate_similarity_by_words(string1, string2)
    return similarity >= threshold


def calculate_similarity_by_words(string1, string2):
    """
    This function takes two strings and calculates their similarity based on the number of common
    words.

    :param string1: First string to compare
    :param string2: Second string to compare
    """

    # Tokenization and normalization: convert to lowercase, remove leading/trailing spaces and split
    # into words
    words1 = set(string1.lower().strip().split())
    words2 = set(string2.lower().strip().split())

    # Find common words
    common_words = words1.intersection(words2)

    # Calculate the similarity as the ratio of common words to total unique words in both strings
    total_words = words1.union(words2)
    similarity = len(common_words) / len(total_words) if total_words else 0

    return similarity


def get_top50_institute_flag(InstituteList, top_50_universities):
    """
    This function takes a list of institutes (as a string separated by ';') and the dataframe of the
    top 50 universities,
    and returns a boolean flag indicating whether at least one of the institutes in the list is in
    the top 50 universities.

    :param InstituteList: List of institutes as a string separated by ';'
    :param top_50_universities: Dataframe of the top 50 universities with cleaned names and acronyms
    """

    if InstituteList is None or InstituteList == "" or not isinstance(InstituteList, str):
        return False
    institutes = InstituteList.split(";")

    institutes = list(set(institutes))

    for uni in institutes:
        if is_in_top50_institutes(uni, top_50_universities):
            return True
    return False


def build_windowed_dataset(
    initialPanel, T, lastYear=2024, firstDecisionYear=2010, maxStartingAge=2
):
    """
    This function takes the initial panel with all the companies and their growth stages at
    different ages,
    and creates a new dataset with a time window of T years to determine the target variable.
    The target variable is defined as whether the company will reach the next growth stage within T
    years from the StartingAge
    (the age at which the company was in Early stage for the first time).
    The function returns a new dataset with only the rows where Age is equal to StartingAge, and
    with the target variable defined as described above.

    :param initialPanel: Panel with all the companies and their growth stages at different ages
    :param T: The prediction horizon in years. It has no default: the value lives in
        ``config.yaml`` under ``T``, and every caller passes it from there.
    :param lastYear: Last available year in the dataset (default is 2024)
    :param firstDecisionYear: First decision year kept (default is 2010)
    :param maxStartingAge: Oldest StartingAge kept (default is 2): a firm reaching the early
        stage later would be predicted from a very different point in its life
    """

    # selection of companies that were in Early stage at least once in their history, and their age
    # at that time
    df_preseed = initialPanel.filter(pl.col("GrowthStageGroup").eq("Early").any().over("CompanyID"))

    # drop rows with null GrowthStageGroup, as they cannot be used to determine the target variable
    df_preseed = df_preseed.drop_nulls(subset=["GrowthStageGroup"])

    df_preseed = df_preseed.select(["CompanyID", "Age"])

    # for each company, determine the StartingAge (minimum Age), LastAge (maximum Age) and TargetAge
    # (the age at which we want to predict the growth stage, which is the minimum between
    # StartingAge + T and LastAge)
    df_combined = (
        df_preseed.group_by("CompanyID")
        .agg(
            [
                pl.col("Age").min().alias("StartingAge"),
                pl.col("Age").max().alias("LastAge"),
            ]
        )
        .with_columns(
            pl.min_horizontal(pl.col("StartingAge") + T, pl.col("LastAge")).alias("TargetAge")
        )
        # Age filter: consider only companies that were in Early stage at maxStartingAge or less
        .filter(pl.col("StartingAge") <= maxStartingAge)
    )

    # Join the initial panel with the combined dataframe to have StartingAge, LastAge and TargetAge
    # for each CompanyID
    initialPanel = initialPanel.join(df_combined, on="CompanyID")

    # Select the rows where Age is equal to TargetAge, and keep only the relevant columns for the
    # target variable
    # (GrowthStageGroup at TargetAge, GrowthNextStageGroup at TargetAge, TimeNextStageGroup at
    # TargetAge)

    df_target = initialPanel.filter(pl.col("Age") == pl.col("TargetAge")).select(
        [
            "CompanyID",
            "StartingAge",
            "TargetAge",
            "GrowthStageGroup",
            "TimeNextStageGroup",
            "GrowthNextStageGroup",
        ]
    )

    # Define the target variable: if the company reaches the next growth stage within T years from
    # StartingAge,
    # the target is GrowthNextStageGroup, otherwise it is GrowthStageGroup

    df_target = df_target.with_columns(
        pl.when(pl.col("TargetAge") + pl.col("TimeNextStageGroup") <= pl.col("StartingAge") + T)
        .then(pl.col("GrowthNextStageGroup"))
        .otherwise(pl.col("GrowthStageGroup"))
        .alias("Target")
    )

    # Join the target variable with the initial panel to have all the features for the companies at
    # the TargetAge
    df_target_panel = initialPanel.join(df_target.select(["CompanyID", "Target"]), on="CompanyID")

    # Select the rows where Age is equal to StartingAge, and keep only the relevant columns for the
    # features
    datasetWithTimeWindow = df_target_panel.filter(
        (pl.col("Age") == pl.col("StartingAge"))
        & (pl.col("YearFounded") + pl.col("Age") <= lastYear - T)
        & (pl.col("YearFounded") + pl.col("Age") >= firstDecisionYear)
    )

    return datasetWithTimeWindow


def build_full_history_dataset(initialPanel, dataset_controlled):
    """
    This function takes the initial panel with all the companies and their growth stages at
    different ages,
    and the dataset with time window, and creates a new dataset without time window using the same
    companies as in the dataset with time window.

    :param initialPanel: Panel with all the companies and their growth stages at different ages
    :param dataset_controlled: Dataset with time window, used to select the companies to
        include in the dataset without time window
    (we want to include only the companies that are in the dataset with time window, to make the two
    datasets comparable)
    """

    ids_prev = dataset_controlled["CompanyID"].to_list()

    df_no_tw = (
        initialPanel.sort(["CompanyID", "Age"])
        .filter(
            pl.col("CompanyID").is_in(ids_prev)
            & pl.col("GrowthStageGroup").is_not_null()
            & pl.col("GrowthNextStageGroup").is_not_null()
            & pl.col("Total_People").is_not_null()
        )
        .group_by("CompanyID")
        .agg(
            [
                pl.all()
                .exclude(
                    [
                        "TotalRaised",
                        "UndisclosedAmountShare",
                        "WorkExp_Idx_Mean",
                        "Highest_Degree_Mean",
                        "Avg_Earliest_Year",
                    ]
                )
                .sort_by("Age")
                .last(),
                # Capital adds up over the firm's life, so the full history is
                # the sum of the yearly amounts.
                pl.col("TotalRaised").sum(),
                # A share is not an amount: summing it would mean nothing and
                # keeping its last year would ignore the rest, so it is averaged
                # over the firm's years. It says how much of the capital raised
                # was never disclosed, which is how reliable TotalRaised is.
                pl.col("UndisclosedAmountShare").mean(),
                pl.col("WorkExp_Idx_Mean").mean(),
                pl.col("Highest_Degree_Mean").mean(),
                pl.col("Avg_Earliest_Year").mean(),
            ]
        )
    )

    # Every firm of the windowed dataset has a usable year here: its starting row has a stage and a
    # team, and neither depends on the timing switch. A firm without one means the two panels do
    # not come from the same extraction, and dropping it would make the two datasets describe
    # different firms.
    missing = dataset_controlled.select("CompanyID").join(df_no_tw, on="CompanyID", how="anti")
    if missing.height:
        raise ValueError(
            f"{missing.height} firms of the windowed dataset have no year with a stage, a next "
            "stage and a team in the full-history panel"
        )
    # Same firms in the same order as the windowed dataset: the split is drawn on row positions.
    df_no_tw = dataset_controlled.select("CompanyID").join(
        df_no_tw, on="CompanyID", how="left", maintain_order="left"
    )

    df_target_final = df_no_tw.rename({"GrowthNextStageGroup": "Target"})

    # A firm with no competitor has a count of zero, not a missing value.
    df_target_final = df_target_final.with_columns(
        pl.col("N_Competitors").fill_null(0),
        pl.col("N_Similar").fill_null(0),
        pl.col("Same_Country").fill_null(0),
        pl.col("SimilarityScoreMean").fill_null(0.0),
    )

    return df_target_final


def build_processed_datasets(
    timed_panel,
    snapshot_panel,
    university_ranking_path,
    *,
    T,
    last_year=2024,
    first_decision_year=2010,
    max_starting_age=2,
    missing_threshold=0.5,
    max_missing_per_row=6,
):
    """Build the two datasets the experiments compare, from the two panels.

    The bias-controlled one comes from the **timed** panel, where every attribute
    is the one of the row's own year, and every feature is read at the age the firm
    first reached an early stage. The biased one comes from the **snapshot** panel,
    where the attributes are the ones declared at extraction time, and every
    feature is cumulated over the firm's whole observed life.

    The order is not an implementation detail: the second is built against the
    first and then reduced to its rows and columns, because every comparison of
    the paper assumes the two carry the same firms described in two ways.

    The switches change values, never rows, so the two panels have to carry the
    same firm-years in the same order. Swapping the target between the two datasets
    on ``CompanyID`` depends on it, so it is checked rather than assumed.

    :param timed_panel: The panel built with the timing on.
    :param snapshot_panel: The panel built with the timing off.
    :param university_ranking_path: Path of the raw QS world ranking file.
    :param T: The prediction horizon in years, from ``config.yaml``.
    :param last_year: Last year the extraction covers.
    :param first_decision_year: First decision year the windowed dataset keeps.
    :param max_starting_age: Oldest starting age the windowed dataset keeps.
    :param missing_threshold: Share of missing values past which a column is dropped.
    :param max_missing_per_row: A row with this many missing values or more is dropped.
    :return: The windowed dataset and the full-history one.
    :raises ValueError: If the two panels do not carry the same firm-years.
    """
    keys = ["CompanyID", "Age"]
    if not timed_panel.select(keys).equals(snapshot_panel.select(keys)):
        raise ValueError(
            "the two panels do not carry the same firm-years: rebuild them from the "
            "same extraction, flipping only the switches"
        )
    thresholds = {
        "missing_threshold": missing_threshold,
        "max_missing_per_row": max_missing_per_row,
    }
    controlled = preprocess_dataset(
        build_windowed_dataset(timed_panel, T, last_year, first_decision_year, max_starting_age),
        university_ranking_path,
        **thresholds,
    )
    leakboth = preprocess_dataset(
        build_full_history_dataset(snapshot_panel, controlled),
        university_ranking_path,
        flag_no_time_window=True,
        **thresholds,
    )
    # The rows are already those of the first, in its order: build_full_history_dataset
    # guarantees it, and the preprocessing of the second drops no row.
    return controlled, leakboth.select(controlled.columns)


def create_has_top50_institute_flag(dataset, university_ranking_path):
    """
    This function takes the dataset and the path of the raw QS world university ranking file,
    and creates a new boolean column "HasTop50Institute" that indicates whether the "Institute"
    column contains a top 50 university.

    :param dataset: Input dataset
    :param university_ranking_path: Path of the raw QS world ranking file
    """

    # Load the list of top 50 universities from the raw university ranking file
    top_50 = process_university_list(university_ranking_path)

    # Create a new boolean column "HasTop50Institute" that indicates whether the "Institute" column
    # contains a top 50 university
    datasetWithUniversityFlag = dataset.with_columns(
        pl.col("Institute")
        .map_elements(
            lambda x: get_top50_institute_flag(x, top_50), return_dtype=pl.Boolean, skip_nulls=False
        )
        .alias("HasTop50Institute")
    )

    # Drop the "Institute" column: the flag is what the models read
    datasetWithUniversityFlag = datasetWithUniversityFlag.drop("Institute")

    return datasetWithUniversityFlag


def handle_missing_values(
    dataset, flag_no_time_window, *, missing_threshold=0.5, max_missing_per_row=6
):
    """
    This function takes the dataset and handles the missing values by imputing the competitor
    variables (N_Competitors, N_Similar, Same_Country, SimilarityScoreMean) with 0, and keeping
    the remaining missing values as they are to be handled by the final imputation strategy.
    For the dataset with time window only, it also drops the rows with a missing Total_People,
    the columns missing on missing_threshold of the rows or more, and the rows missing
    max_missing_per_row values or more. The dataset without time window drops nothing: it
    inherits the rows and columns of the first.


    :param dataset: Input dataset
    :param flag_no_time_window: Flag indicating whether to create a time window dataset
    :param missing_threshold: Share of missing values past which a column is dropped
    :param max_missing_per_row: A row with this many missing values or more is dropped
    """

    # Missing values handling

    if not flag_no_time_window:
        # Check if we are creating the time window dataset and if not, keep all the rows (we must
        # keep the same rows in all the experiments to make them comparable)

        # Drop rows with missing values in the 'Total_People' column, as it's a crucial variable for
        # the analysis
        datasetWithNoMissingValues = dataset.drop_nulls(subset=["Total_People"])
    else:
        datasetWithNoMissingValues = dataset.clone()

    # Convert boolean columns to integers (0 and 1)
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col(pl.Boolean).cast(pl.Int8)
    )

    # For these variables, it was decided to impute missing values to 0
    cols_toInt = ["N_Competitors", "N_Similar", "Same_Country"]
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        [pl.col(name).cast(pl.Int64).fill_null(0) for name in cols_toInt]
    )

    # A missing mean similarity means the firm has no comparable company, which is
    # the same thing a missing count means, so it takes the same zero. The mean of
    # the column would be a statistic of the whole sample -- other firms, later
    # years and the test rows included -- imputed into a single row.
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col("SimilarityScoreMean").fill_null(0.0)
    )

    # For the remaining variables, we keep the missing values as they are, since they will be
    # handled by the final imputation strategy
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col(
            [
                "YearFounded",
                "Age",
                "Total_Founders",
                "Total_People",
                "Is_Eco",
                "Is_Eng",
                "Is_NS",
                "Is_Hum",
                "Is_SS",
                "Is_Med",
                "Is_Law",
                "Is_IT",
            ]
        ).cast(pl.Int64)
    )

    # Check if we are creating the time window dataset and if not, keep all the columns (we must
    # keep the same columns in all the experiments to make them comparable)
    # otherwise select only those that meet the missing values threshold
    if not flag_no_time_window:
        # Feature selection based on missing values threshold

        # A feature missing on missing_threshold of the rows or more is dropped:
        # past that point the imputation would be inventing the column rather
        # than completing it. It comes from config.yaml, and the command line can
        # override it for one run.
        threshold = missing_threshold

        # Identify columns to keep based on the proportion of missing values
        cols_to_keep = [
            col
            for col in datasetWithNoMissingValues.columns
            if datasetWithNoMissingValues[col].null_count() / len(datasetWithNoMissingValues)
            < threshold
        ]

        datasetWithNoMissingValues = datasetWithNoMissingValues.select(cols_to_keep)

    # row selection based on missing values threshold
    if not flag_no_time_window:
        # Check if we are creating the time window dataset and if not, keep all the rows (we must
        # keep the same rows in all the experiments to make them comparable)
        # otherwise drop the rows that are mostly empty.

        # Count missing values per row and analyze the distribution
        datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
            null_count_row=pl.sum_horizontal(pl.all().is_null())
        )

        datasetWithNoMissingValues = datasetWithNoMissingValues.filter(
            pl.col("null_count_row") < max_missing_per_row
        ).drop("null_count_row")

    return datasetWithNoMissingValues


def preprocess_dataset(
    dataset,
    university_ranking_path,
    flag_no_time_window=False,
    *,
    missing_threshold=0.5,
    max_missing_per_row=6,
):
    """
    This function takes the initial dataset and the path of the raw QS world university ranking
    file, and performs the following preprocessing steps:
    - Select only the relevant columns for the features and the target variable
    - Create the "HasTop50Institute" flag based on the "Institute" column and the top 50
      universities list
    - Turn "Gender_CEO" into a single indicator column. HQCountry and
      PrimaryIndustrySector are left as raw categories: they are frequency-encoded per
      split, in src/encoding.py, so that a held-out row never contributes to its own
      encoding
    - Handle missing values with handle_missing_values: the competitor variables are imputed
      with 0 and the remaining missing values are left to the final imputation strategy; the
      dataset with time window also drops the rows and columns that are mostly empty.
    - Create the final dataset by encoding the target variable as binary (1 for "Later" and "Exit"
      (Success), 0 for "Early" and "Out")

    :param dataset: Input dataset
    :param university_ranking_path: Path of the raw QS world ranking file
    :param flag_no_time_window: Flag indicating whether to create a time window dataset
    (default is False, meaning that we will create the time window dataset, if True we will create
    the dataset without time window)
    :param missing_threshold: Share of missing values past which a column is dropped
    :param max_missing_per_row: A row with this many missing values or more is dropped
    """

    # Select only the relevant columns for the features and the target variable
    filteredDataset = dataset.select(["CompanyID", "Target", *FEATURE_COLUMNS])

    # Create the "HasTop50Institute" flag based on the "Institute" column and the top 50
    # universities list
    filteredDatasetWithUniversityFlag = create_has_top50_institute_flag(
        filteredDataset, university_ranking_path
    )

    # Handle Gender_CEO categorical variable in binary format: one indicator, and
    # the other two levels dropped.
    filteredDatasetWithUniversityFlag = filteredDatasetWithUniversityFlag.to_dummies("Gender_CEO")
    spent = ["Gender_CEO_Male", "Gender_CEO_null"]
    filteredDatasetWithUniversityFlag = filteredDatasetWithUniversityFlag.drop(
        [c for c in spent if c in filteredDatasetWithUniversityFlag.columns]
    )
    if "Gender_CEO_Female" not in filteredDatasetWithUniversityFlag.columns:
        filteredDatasetWithUniversityFlag = filteredDatasetWithUniversityFlag.with_columns(
            pl.lit(0, dtype=pl.UInt8).alias("Gender_CEO_Female")
        )

    # Handle missing values
    # Only the dataset with time window drops rows and columns for their missing values. The one
    # without time window drops nothing here: its rows are the firms of the first by
    # construction (build_full_history_dataset), and build_processed_datasets gives it the
    # columns of the first, so its own missing values never decide what is kept.
    # What they still decide is what the final imputation has to fill: a row or a column kept
    # because the first dataset kept it can be emptier here than the thresholds would allow.
    # With a different starting dataset, check how many missing values are left in the dataset
    # without time window before trusting the imputation with them.
    datasetWithNoMissingValues = handle_missing_values(
        filteredDatasetWithUniversityFlag,
        flag_no_time_window,
        missing_threshold=missing_threshold,
        max_missing_per_row=max_missing_per_row,
    )

    # Create the final dataset by encoding the target variable as binary (1 for "Later" and "Exit"
    # (Success), 0 for "Early" and "Out")
    finalDataset = datasetWithNoMissingValues.with_columns(
        pl.when(pl.col("Target").is_in(["Later", "Exit"])).then(1).otherwise(0).alias("Target")
    )

    return finalDataset
