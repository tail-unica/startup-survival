import pandas as pd
import polars as pl
import re


def processUniversityList(path):


    df_uni = pd.read_csv(path)


    df_uni.columns = [col.strip().replace(' ', '_') for col in df_uni.columns]

    df_uni['Overall_Score'] = pd.to_numeric(df_uni['Overall_Score'], errors='coerce')
    df_uni_filtered = df_uni.dropna(subset=['Overall_Score'])

    # Ordinamento prima per Università e poi per Anno (dal più recente al più vecchio)
    df_uni_sorted = df_uni_filtered.sort_values(by=['University', 'Year'], ascending=[True, False])

    # Estrazione della prima riga per ogni università (essendo ordinate per anno, sarà la più recente)
    result_df_uni = df_uni_sorted.drop_duplicates(subset=['University'], keep='first')

    result_df_uni=result_df_uni.sort_values(by='Overall_Score',ascending=False)

    #Selezione delle colonne finali
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
    #Estrazione acronimo, se presente
    top_50_universities["acronym"] = top_50_universities["University"].str.lower().str.extract(r"\((.*?)\)")

    return top_50_universities




def IsInTop50Institutes(university,top_50_universities):

    top50nomi=top_50_universities["Cleaned_University"].dropna().unique()
    top50acronimi=top_50_universities["acronym"].dropna().unique()

    match = re.search(r"\((.*?)\)", str(university).lower())
    if match:
        acrP = match.group(1)
    else:
        acrP = None
    
    uniP = str(university).lower()
    uniP = re.sub(r"\(.*?\)", "", uniP)  # rimozione il testo tra parentesi
    uniP = uniP.replace("university", "") \
            .replace("of", "") \
            .replace("the", "") \
            .replace(",", "") \
            .replace("-", "")

    uniP = re.sub(r"\s+", " ", uniP)  # rimozione gli spazi multipli
    uniP = uniP.strip()                # rimozione spazi iniziali e finali


    if not isinstance(uniP, str):
        uniP = ""
    if not isinstance(acrP, str):
        acrP = ""
    
    if(acrP!="" and acrP in top50acronimi):
        return True
    
    if(uniP!=""):
        if(uniP in top50nomi):
            return True

        for acronimo in top50acronimi:
            if not isinstance(acronimo, str):
                acronimo = ""
            if(acronimo !="" and acronimo in uniP):
                return True
        for nome in top50nomi:
            if(nome in uniP or are_strings_similar(nome,uniP)):
                return True
    return False

def are_strings_similar(string1, string2, threshold=0.5):
    similarity = calculate_similarity_by_words(string1, string2)
    return similarity >= threshold

def calculate_similarity_by_words(string1, string2):
    # Tokenizzazione e normalizzazione (lowercase e rimozione spazi extra)
    words1 = set(string1.lower().strip().split())
    words2 = set(string2.lower().strip().split())
    
    # Ricerca delle parole comuni
    common_words = words1.intersection(words2)
    
    # Calcolo della similarità
    total_words = words1.union(words2)
    similarity = len(common_words) / len(total_words) if total_words else 0
    
    return similarity

def getFlagTop50Institute(InstituteList,top_50_universities):
    if InstituteList==None or InstituteList=='' or not isinstance(InstituteList, str):
        return False
    lista = InstituteList.split(';')

    lista=list(set(lista))

    for uni in lista:
        if(IsInTop50Institutes(uni,top_50_universities)):
            return True
    return False

def lump_categories(df, column_name, min_count, other_label="Other"):
    return df.with_columns(
        pl.when(pl.col(column_name).count().over(column_name) >= min_count)
        .then(pl.col(column_name))
        .otherwise(pl.lit(other_label))
        .alias(column_name)
    )

