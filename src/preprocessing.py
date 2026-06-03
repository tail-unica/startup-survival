import pandas as pd
import polars as pl
import re


def processUniversityList(path):

    '''
    This function takes the path of the raw QS world university ranking file, and returns a dataframe with the top 50 universities, 
    their overall score and the year of the ranking. The function also cleans the university names by removing common words and punctuation, 
    and extracts the acronym if present.
    
    :param path: Path of the raw QS world ranking file
    '''
    df_uni = pd.read_csv(path)


    df_uni.columns = [col.strip().replace(' ', '_') for col in df_uni.columns]

    df_uni['Overall_Score'] = pd.to_numeric(df_uni['Overall_Score'], errors='coerce')
    df_uni_filtered = df_uni.dropna(subset=['Overall_Score'])

    # Sort the dataframe by University and Year in descending order
    df_uni_sorted = df_uni_filtered.sort_values(by=['University', 'Year'], ascending=[True, False])

    # Keep only the first occurrence of each university 
    result_df_uni = df_uni_sorted.drop_duplicates(subset=['University'], keep='first')

    result_df_uni=result_df_uni.sort_values(by='Overall_Score',ascending=False)

    final_result_uni = result_df_uni[['University', 'Overall_Score', 'Year']]

    top_50_universities=final_result_uni.head(50).copy()

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
    top_50_universities["acronym"] = top_50_universities["University"].str.lower().str.extract(r"\((.*?)\)")

    return top_50_universities




def IsInTop50Institutes(university,top_50_universities):

    '''
    This function takes a university name and the dataframe of the top 50 universities, 
    and returns a boolean flag indicating whether the university is in the top 50 universities.
    
    :param university: University name to check
    :param top_50_universities: Dataframe of the top 50 universities with cleaned names and acronyms
    '''

    top50names=top_50_universities["Cleaned_University"].dropna().unique()
    top50acronyms=top_50_universities["acronym"].dropna().unique()

    match = re.search(r"\((.*?)\)", str(university).lower())
    if match:
        acrP = match.group(1)
    else:
        acrP = None
    
    uniP = str(university).lower()
    uniP = re.sub(r"\(.*?\)", "", uniP)  # revomove text between parentheses
    uniP = uniP.replace("university", "") \
            .replace("of", "") \
            .replace("the", "") \
            .replace(",", "") \
            .replace("-", "")

    uniP = re.sub(r"\s+", " ", uniP)  # replace multiple spaces with a single space
    uniP = uniP.strip()                # remove leading and trailing spaces


    if not isinstance(uniP, str):
        uniP = ""
    if not isinstance(acrP, str):
        acrP = ""
    
    if(acrP!="" and acrP in top50acronyms):
        return True
    
    if(uniP!=""):
        if(uniP in top50names):
            return True

        for acronym in top50acronyms:
            if not isinstance(acronym, str):
                acronym = ""
            if(acronym !="" and acronym in uniP):
                return True
        for name in top50names:
            if(name in uniP or are_strings_similar(name,uniP)):
                return True
    return False

def are_strings_similar(string1, string2, threshold=0.5):

    '''
    This function takes two strings and a similarity threshold, 
    and returns a boolean flag indicating whether the two strings are similar based on the similarity of their words.

    :param string1: First string to compare
    :param string2: Second string to compare
    :param threshold: Similarity threshold (default is 0.5, meaning that the strings are considered similar if at least 50% of their words are common)
    '''

    similarity = calculate_similarity_by_words(string1, string2)
    return similarity >= threshold

def calculate_similarity_by_words(string1, string2):

    '''
    This function takes two strings and calculates their similarity based on the number of common words.
    
    :param string1: First string to compare
    :param string2: Second string to compare
    '''

    # Tokenization and normalization: convert to lowercase, remove leading/trailing spaces and split into words
    words1 = set(string1.lower().strip().split())
    words2 = set(string2.lower().strip().split())
    
    # Find common words
    common_words = words1.intersection(words2)
    
    # Calculate the similarity as the ratio of common words to total unique words in both strings
    total_words = words1.union(words2)
    similarity = len(common_words) / len(total_words) if total_words else 0
    
    return similarity

def getFlagTop50Institute(InstituteList,top_50_universities):

    '''
    This function takes a list of institutes (as a string separated by ';') and the dataframe of the top 50 universities,
    and returns a boolean flag indicating whether at least one of the institutes in the list is in the top 50 universities.
    
    :param InstituteList: List of institutes as a string separated by ';'
    :param top_50_universities: Dataframe of the top 50 universities with cleaned names and acronyms
    '''

    if InstituteList==None or InstituteList=='' or not isinstance(InstituteList, str):
        return False
    lista = InstituteList.split(';')

    lista=list(set(lista))

    for uni in lista:
        if(IsInTop50Institutes(uni,top_50_universities)):
            return True
    return False

def collapse_categories(df, column_name, min_count, other_label="Other"):

    '''
    This function takes a dataframe, a column name, a minimum count and an other label,
    and returns a new dataframe where the categories in the specified column that have a count less than the minimum count are replaced with the other label.
    
    :param df: Input dataframe
    :param column_name: Name of the column to collapse categories
    :param min_count: Minimum count for a category to be kept as is, otherwise it will be replaced with the other label
    :param other_label: Label to replace the categories that have a count less than the minimum count (default is "Other")
    '''

    return df.with_columns(
        pl.when(pl.col(column_name).count().over(column_name) >= min_count)
        .then(pl.col(column_name))
        .otherwise(pl.lit(other_label))
        .alias(column_name)
    )

def getCompleteDatasetWithTimeWindow(initialPanel,TimeWindow=7,lastYear=2024):

    '''
    This function takes the initial panel with all the companies and their growth stages at different ages, 
    and creates a new dataset with a time window of T years to determine the target variable. 
    The target variable is defined as whether the company will reach the next growth stage within T years from the StartingAge 
    (the age at which the company was in Early stage for the first time). 
    The function returns a new dataset with only the rows where Age is equal to StartingAge, and with the target variable defined as described above.

    :param initialPanel: Panel with all the companies and their growth stages at different ages
    :param TimeWindow: Time window in years to determine the target variable (default is 7 years, as we want to predict if the company will reach the next growth stage within 7 years from the StartingAge)
    :param lastYear: Last available year in the dataset (default is 2024)
    '''

    # selection of companies that were in Early stage at least once in their history, and their age at that time 
    df_preseed = initialPanel.filter(
        pl.col("GrowthStageGroup")
        .eq("Early")
        .any()
        .over("CompanyID")
    )

    # drop rows with null GrowthStageGroup, as they cannot be used to determine the target variable
    df_preseed=df_preseed.drop_nulls(subset=['GrowthStageGroup'])


    df_preseed = df_preseed.select(["CompanyID", "Age"])

    # for each company, determine the StartingAge (minimum Age), LastAge (maximum Age) and TargetAge 
    # (the age at which we want to predict the growth stage, which is the minimum between StartingAge + T and LastAge)
    df_combined = (
        df_preseed
        .group_by("CompanyID")
        .agg([
            pl.col("Age").min().alias("StartingAge"),
            pl.col("Age").max().alias("LastAge"),
        ])
        .with_columns(
            pl.min_horizontal(
                pl.col("StartingAge") + TimeWindow,
                pl.col("LastAge")
            ).alias("TargetAge")
        )
        .filter(pl.col("StartingAge") <= 2) #Age filter: consider only companies that were in Early stage at age 2 or less
    )

    # Join the initial panel with the combined dataframe to have StartingAge, LastAge and TargetAge for each CompanyID
    initialPanel = initialPanel.join(df_combined, on="CompanyID")

    # Select the rows where Age is equal to TargetAge, and keep only the relevant columns for the target variable 
    # (GrowthStageGroup at TargetAge, GrowthNextStageGroup at TargetAge, TimeNextStageGroup at TargetAge)

    df_target = initialPanel.filter(
        pl.col("Age") == pl.col("TargetAge")
    ).select(["CompanyID", "StartingAge" ,"TargetAge" ,'GrowthStageGroup','TimeNextStageGroup','GrowthNextStageGroup'])

    # Define the target variable: if the company reaches the next growth stage within T years from StartingAge,
    # the target is GrowthNextStageGroup, otherwise it is GrowthStageGroup

    df_target = df_target.with_columns(
        pl.when(pl.col("TargetAge") + pl.col("TimeNextStageGroup")<=pl.col("StartingAge") + TimeWindow) 
        .then(pl.col("GrowthNextStageGroup"))
        .otherwise(pl.col("GrowthStageGroup"))
        .alias("Target")
    )

    # Join the target variable with the initial panel to have all the features for the companies at the TargetAge
    df_target_panel=initialPanel.join(df_target.select(["CompanyID", "Target"]),on='CompanyID')

    # Select the rows where Age is equal to StartingAge, and keep only the relevant columns for the features
    datasetWithTimeWindow = df_target_panel.filter(
        (pl.col("Age") == pl.col("StartingAge")) &
        (pl.col("YearFounded")+pl.col("Age") <= lastYear-TimeWindow) &
        (pl.col("YearFounded")+pl.col("Age") >= 2010)
    )

    datasetWithTimeWindow=datasetWithTimeWindow.drop("N_Competitors_All", "Same_Country_All", "SimilarityScoreMean_All")

    return datasetWithTimeWindow

def getCompleteDatasetWithoutTimeWindow(initialPanel, dataset_window):


    """
    This function takes the initial panel with all the companies and their growth stages at different ages,
    and the dataset with time window, and creates a new dataset without time window using the same companies as in the dataset with time window.
    
    :param initialPanel: Panel with all the companies and their growth stages at different ages
    :param dataset_window: Dataset with time window, used to select the companies to include in the dataset without time window 
    (we want to include only the companies that are in the dataset with time window, to make the two datasets comparable)
    """


    ids_prev=dataset_window["CompanyID"].to_list()

    df_no_tw = (
        initialPanel
        .sort(["CompanyID", "Age"])
        .filter(
            pl.col("CompanyID").is_in(ids_prev) &
            pl.col("GrowthStageGroup").is_not_null() &
            pl.col("GrowthNextStageGroup").is_not_null() &
            pl.col("Total_People").is_not_null()
        )
        .group_by("CompanyID")
        .agg([
            pl.all().exclude(["TotalRaised_Est", "WorkExp_Idx_Mean", "Highest_Degree_Mean",'Avg_Earliest_Year'])
            .sort_by("Age").last(),
            pl.col("TotalRaised_Est").sum(),
            pl.col("WorkExp_Idx_Mean").mean(),
            pl.col("Highest_Degree_Mean").mean(),
            pl.col("Avg_Earliest_Year").mean(),
        ])
    )

    df_target_final = df_no_tw.rename({"GrowthNextStageGroup": "Target"})

    df_target_final = (
        df_target_final
        .drop("N_Competitors", "Same_Country", "SimilarityScoreMean")
        .rename({
            "N_Competitors_All": "N_Competitors",
            "Same_Country_All": "Same_Country",
            "SimilarityScoreMean_All": "SimilarityScoreMean",
        })
        .with_columns(
            pl.col("N_Competitors").fill_null(0),
            pl.col("Same_Country").fill_null(0),
            pl.col("SimilarityScoreMean").fill_null(0.0),
        )
    )

    return df_target_final


def createHasTop50InstituteFlag(dataset, university_ranking_path):

    '''
    This function takes the dataset and the path of the raw QS world university ranking file, 
    and creates a new boolean column "HasTop50Institute" that indicates whether the "Institute" column contains a top 50 university.
    
    :param dataset: Input dataset
    :param university_ranking_path: Path of the raw QS world ranking file
    '''

    # Load the list of top 50 universities from the raw university ranking file
    top_50 = processUniversityList(university_ranking_path)

    # Create a new boolean column "HasTop50Institute" that indicates whether the "Institute" column contains a top 50 university
    datasetWithUniversityFlag=dataset.with_columns(pl.col("Institute").map_elements( lambda x: getFlagTop50Institute(x, top_50),return_dtype=pl.Boolean, 
            skip_nulls=False).alias("HasTop50Institute"))

    # Drop the original "Institute" column as it's no longer needed
    datasetWithUniversityFlag=datasetWithUniversityFlag.drop("Institute")

    return datasetWithUniversityFlag

def handleCategoricalVariables(dataset):

    '''
    This function takes the dataset and handles the categorical variables by collapsing categories with low frequency into "Other" 
    and applying frequency encoding.
    
    :param dataset: Input dataset
    '''

    # collapse categories in HQCountry with less than 1000 occurrences into "Other"
    datasetWithNoCategories = collapse_categories(dataset, "HQCountry", min_count=1000)


    #frequency encoding HQCountry
    freq_df = datasetWithNoCategories.group_by("HQCountry").agg(
        pl.len().alias("HQCountryFreq")
    )
    datasetWithNoCategories = datasetWithNoCategories.join(freq_df, on="HQCountry").drop("HQCountry")


    #frequency encoding PrimaryIndustrySector
    freq_df = datasetWithNoCategories.group_by("PrimaryIndustrySector").agg(
        pl.len().alias("PrimaryIndustrySectorFreq")
    )
    datasetWithNoCategories = datasetWithNoCategories.join(freq_df, on="PrimaryIndustrySector").drop("PrimaryIndustrySector")


    # indicator feature for the CEO's gender
    datasetWithNoCategories = datasetWithNoCategories.to_dummies("Gender_CEO").drop("Gender_CEO_Male").drop("Gender_CEO_null")

    return datasetWithNoCategories

def handleMissingValues(dataset, flag_no_time_window):

    '''
    This function takes the dataset and handles the missing values by dropping rows with missing values in crucial variables, 
    imputing missing values in some variables with 0, imputing missing values in other variables with the mean of the column, 
    and keeping the remaining missing values as they are to be handled by the final imputation strategy. 
    The function also applies a feature selection based on the proportion of missing values in each column, 
    and a row selection based on the number of missing values in each row.
    
    
    :param dataset: Input dataset
    :param flag_no_time_window: Flag indicating whether to create a time window dataset
    '''

    #Missing values handling

    if(not flag_no_time_window):
        # Check if we are creating the time window dataset and if not, keep all the rows (we must keep the same rows in all the experiments to make them comparable)

        # Drop rows with missing values in the 'Total_People' column, as it's a crucial variable for the analysis
        datasetWithNoMissingValues = dataset.drop_nulls(subset=['Total_People'])
    else:
        datasetWithNoMissingValues = dataset.clone()

    # Convert boolean columns to integers (0 and 1)
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col(pl.Boolean).cast(pl.Int8)
    )

    # For these variables, it was decided to impute missing values to 0
    variabili_toInt = [ 'N_Competitors','Same_Country'] 
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        [pl.col(name).cast(pl.Int64).fill_null(0) for name in variabili_toInt]
    )

    # Impute missing values in 'SimilarityScoreMean' with the mean of the column
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col('SimilarityScoreMean').fill_null(pl.col('SimilarityScoreMean').mean()) 
    )

    # For the remaining variables, we keep the missing values as they are, since they will be handled by the final imputation strategy 
    datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
        pl.col(["YearFounded","Age",'Total_Founders','Total_People','Is_Eco','Is_Eng','Is_NS','Is_Hum','Is_SS','Is_Med','Is_Law','Is_IT']).cast(pl.Int64)
    )


    # Check if we are creating the time window dataset and if not, keep all the columns (we must keep the same columns in all the experiments to make them comparable)
    # otherwise select only those that meet the missing values threshold
    if(not flag_no_time_window):

        # Feature selection based on missing values threshold

        threshold = 0.4

        # Identify columns to keep based on the proportion of missing values
        cols_to_keep = [
            col for col in datasetWithNoMissingValues.columns 
            if datasetWithNoMissingValues[col].null_count() / len(datasetWithNoMissingValues) < threshold
        ]

        datasetWithNoMissingValues = datasetWithNoMissingValues.select(cols_to_keep)


    # row selection based on missing values threshold
    if(not flag_no_time_window):

        # Check if we are creating the time window dataset and if not, keep all the rows (we must keep the same rows in all the experiments to make them comparable)
        # otherwise drop those with 4 or more missing values 

        # Count missing values per row and analyze the distribution
        datasetWithNoMissingValues = datasetWithNoMissingValues.with_columns(
            null_count_row = pl.sum_horizontal(pl.all().is_null())
        )


        datasetWithNoMissingValues = datasetWithNoMissingValues.filter(pl.col("null_count_row") < 4).drop("null_count_row")
        

    return datasetWithNoMissingValues


def preprocessDataset(dataset, university_ranking_path, flag_no_time_window=False):
  
    '''
    This function takes the initial dataset and the path of the raw QS world university ranking file, and performs the following preprocessing steps:
    - Select only the relevant columns for the features and the target variable
    - Create the "HasTop50Institute" flag based on the "Institute" column and the top 50 universities list
    - Handle categorical variables by collapsing low frequency categories and applying frequency encoding
    - Handle missing values by dropping rows with missing values in crucial variables, imputing missing values in some variables with 0,
      imputing missing values in other variables with the mean of the column, and keeping the remaining missing values as they are to 
      be handled by the final imputation strategy. The function also applies a feature selection based on the proportion of missing values in each column, 
      and a row selection based on the number of missing values in each row.
    - Create the final dataset by encoding the target variable as binary (1 for "Later" and "Exit"  (Success), 0 for "Steady")      

    :param dataset: Input dataset
    :param university_ranking_path: Path of the raw QS world ranking file
    :param flag_no_time_window: Flag indicating whether to create a time window dataset 
    (default is False, meaning that we will create the time window dataset, if True we will create the dataset without time window)
    '''


    # Select only the relevant columns for the features and the target variable
    filteredDataset=dataset.select(['CompanyID', 'Target' ,'YearFounded','Age', 'N_Deal', 'TotalRaised_Est', 
                                            'Percent_Females', 'Is_Eco', 'Is_Eng', 'Is_NS', 
                                            'Is_Hum', 'Is_SS', 'Is_Med', 'Is_Law', 'Is_IT', 
                                            'Institute', 'WorkExp_Idx_Mean', 'Total_Founders', 'Is_Debt', 
                                            'Is_SpinOff', 'Is_CrowdFunding',  
                                            'MeanMedianRoundAmount_cum', 'Is_Accelerator', 'has_Corporate', 
                                            'has_VentureCapital', 'has_PublicInvestor', 'has_Angel_Lead', 'has_Corporate_Lead',
                                            'has_VentureCapital_Lead', 'has_Accelerator_Lead', 'has_PrivateEquity_Lead', 'has_PublicInvestor_Lead', 
                                                'HQCountry', 'PrimaryIndustrySector',
                                            'SimilarityScoreMean', 'N_Competitors', 'Same_Country','Highest_Degree_CEO','Gender_CEO','MeanTotalInvestments_cum',
                                            'WorkExperienceIndex_CEO', 'Is_Angel',  'Total_People', 'Is_Grant','has_PrivateEquity','TotalInvestors', 'Highest_Degree_Mean','Avg_Earliest_Year'
                                            ])


    # Create the "HasTop50Institute" flag based on the "Institute" column and the top 50 universities list
    filteredDatasetWithUniversityFlag = createHasTop50InstituteFlag(filteredDataset, university_ranking_path)

    # Handle categorical variables by collapsing low frequency categories and applying frequency encoding
    datasetWithNoCategories = handleCategoricalVariables(filteredDatasetWithUniversityFlag)

    # Handle missing values 
    # Keep attention: in these experiments i'm assuming that the missing values are in the same columns for both the dataset with time window and the dataset without time window,
    #  and that the distribution of missing values is similar in the two datasets, so I apply the same missing values handling strategy to both datasets. 
    #If you want to replicate the experiments with a different starting dataset, you should check if these assumptions hold and eventually adapt the missing values handling strategy to the specific characteristics of the new dataset.
    datasetWithNoMissingValues = handleMissingValues(datasetWithNoCategories, flag_no_time_window)
    

    # Create the final dataset by encoding the target variable as binary (1 for "Later" and "Exit"  (Success), 0 for "Steady")
    finalDataset = datasetWithNoMissingValues.with_columns(
        pl.when(pl.col("Target").is_in(["Later", "Exit"]))
        .then(1)
        .otherwise(0)
        .alias("Target")
    )


    return finalDataset