from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFECV
from sklearn.compose import TransformedTargetRegressor

from category_encoders import CatBoostEncoder
# from category_encoders.target_encoder import TargetEncoder

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

from lightgbm.sklearn import LGBMRegressor

import numpy as np
import pandas

# CSV file paths
results_file = "results/tcd-ml-1920-group-income-submission.csv"
test_file = "tcd-ml-1920-group-income-test.csv"
train_file = "tcd-ml-1920-group-income-train.csv"

all_columns = [
    "Instance",
    "Year of Record",
    "Housing Situation",
    "Crime Level in the City of Employement",
    "Work Experience in Current Job [years]",
    "Satisfation with employer",
    "Gender",
    "Age",
    "Country",
    "Size of City",
    "Profession",
    "University Degree",
    "Wears Glasses",
    "Hair Color",
    "Body Height [cm]",
    "Yearly Income in addition to Salary (e.g. Rental Income)"
]
no_income_columns = [
    "Year of Record",
    "Housing Situation",
    "Crime Level in the City of Employement",
    "Work Experience in Current Job [years]",
    "Satisfation with employer",
    "Gender",
    "Age",
    "Country",
    "Size of City",
    "Profession",
    "University Degree",
    "Wears Glasses",
    "Hair Color",
    "Body Height [cm]"
]
target_columns = [
    "Total Yearly Income [EUR]"
]

col_types = {
    # "Year of Record": int,
    "Housing Situation": str,
    # "Crime Level in the City of Employement": str,
    # "Work Experience in Current Job [years]": int,
    # "Satisfation with employer": str,
    # "Gender": str,
    # "Age": int,
    # "Country": str,
    # "Size of City": float,
    # "Profession": str,
    # "University Degree": str,
    # "Wears Glasses": int,
    # "Hair Color": str,
    # "Body Height [cm]": float,
    # "Yearly Income in addition to Salary (e.g. Rental Income)": float
    #  ^ remove "EUR" and coerce to float
}

def reg_model(labelled_data, unlabelled_data):
    """ Parameters: training dataframe, unknown dataframe
        Returns: results dataframe (Instance, Income)

        ffill on NaN from training data,
        Replaces NaN in test data with ffill, 
        cat-encodes non-numeric fields, 
        scales values,
        80/20 splits data to help verify model, 
        uses LightGBM
    """

    # print("throwing away rows to speed up model")
    # speed up testing by throwing away some data
    # clean_labelled = labelled_data.sample(frac=0.2)
    clean_labelled = labelled_data.copy()
    clean_unlabelled = unlabelled_data.copy()
 
    print("cleaning data...")
    # get rid of weird value
    clean_labelled.loc[
        :, "Work Experience in Current Job [years]"
    ] = pandas.to_numeric(
        labelled_data["Work Experience in Current Job [years]"], 
        errors="coerce"
    )
    clean_unlabelled.loc[
        :, "Work Experience in Current Job [years]"
    ] = pandas.to_numeric(
        unlabelled_data["Work Experience in Current Job [years]"], 
        errors="coerce"
    )
    print("mixed type issue fixed..")

    # fix additional income field
    clean_labelled.loc[
        :, "Yearly Income in addition to Salary (e.g. Rental Income)"
    ] = pandas.to_numeric(
        np.fromiter(map(
            lambda s: s.replace(" EUR", ""),
            clean_labelled["Yearly Income in addition to Salary (e.g. Rental Income)"],
        ), dtype=np.float),
        errors="coerce"
    )
    clean_unlabelled.loc[
        :, "Yearly Income in addition to Salary (e.g. Rental Income)"
    ] = pandas.to_numeric(
        np.fromiter(map(
            lambda s: s.replace(" EUR", ""),
            clean_unlabelled["Yearly Income in addition to Salary (e.g. Rental Income)"],
        ), dtype=np.float),
        errors="coerce"
    )

    # dropping useless columns
    drop_columns(clean_unlabelled)
    drop_columns(clean_labelled)

    # removing NaN values
    clean_labelled.fillna(method="ffill", inplace=True)
    clean_unlabelled = clean_unlabelled[all_columns]
    clean_unlabelled.fillna(method="ffill", inplace=True) 

    # input data for final predictions
    unknown_data = clean_unlabelled.drop(["Instance"], axis=1)

    print("splitting data into train and test...")
    # 80/20 split, and separating targets
    split = split_data(clean_labelled)
    train_data, train_target, test_data, test_target = split

    print("encoding categorical data...")
    # categorical encoding
    cat = CatBoostEncoder()
    train_data = cat.fit_transform(train_data, train_target)
    test_data = cat.transform(test_data)
    unknown_data = cat.transform(unknown_data)

    # separate additional income
    train_add_income = train_data[
        "Yearly Income in addition to Salary (e.g. Rental Income)"
    ].values
    test_add_income = test_data[
        "Yearly Income in addition to Salary (e.g. Rental Income)"
    ].values
    unknown_add_income = unknown_data[
        "Yearly Income in addition to Salary (e.g. Rental Income)"
    ].values

    train_data = train_data[no_income_columns]
    test_data = test_data[no_income_columns]
    unknown_data = unknown_data[no_income_columns]

    train_target = train_target["Total Yearly Income [EUR]"].values - train_add_income
    test_target = test_target["Total Yearly Income [EUR]"].values

    print("scaling values...")
    # scaling values
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    unknown_data = scaler.transform(unknown_data)

    print("fitting model...")
    # fit model
    reg = LGBMRegressor()
    # reg = TransformedTargetRegressor(
    #     regressor=mod,
    #     transformer=scaler
    # )
    reg.fit(train_data, train_target)

    print("predicting test data...")
    test_result = reg.predict(test_data, num_iterations=15000)
    # add additional income
    test_result = test_result + test_add_income

    print("analysing test results...")
    # validate test
    error = mean_absolute_error(test_target, test_result)
    score = explained_variance_score(test_target, test_result)
    print("Mean absolute error of test data: ", error)
    print("Score: ", score)

    print("predicting unknown data...")
    # predict and format
    values = reg.predict(unknown_data)
    values = values + unknown_add_income

    results = pandas.DataFrame({
        "Instance": clean_unlabelled["Instance"].values,
        "Total Yearly Income [EUR]": values
    })
    print("Finished.")
    return results


def drop_columns(dataframe):
    """ Remove unnecessary input columns before inserting into model """
    return dataframe.drop(
        [
            "Hair Color",
            "Wears Glasses"
        ],
        axis=1
    )

def feature_selection(train_data, train_target, test_data, unknown_data):
    """ Selects features based on cross validation with Lasso 
        This method determined the above removed columns
        Not calling it everytime, because it takes ages to run
    """
    lasso = Lasso()
    selector = RFECV(lasso, cv=3)

    train = selector.fit_transform(train_data, train_target)
    test = selector.transform(test_data)
    unknown = selector.transform(unknown_data)

    print(selector.support_)    # mask of used and deleted columns
    return (train, test, unknown)

def split_data(dataframe):
    """ Splits data into training and test, also splits input from target"""
    train, test = train_test_split(dataframe, test_size=0.2)

    train_data = train.drop(["Total Yearly Income [EUR]", "Instance"], axis=1)
    train_target = train[target_columns]

    test_data = test.drop(["Total Yearly Income [EUR]", "Instance"], axis=1)
    test_target = test[target_columns]

    return (train_data, train_target, test_data, test_target)


if __name__ == "__main__":
    print("reading files...")
    train_data = pandas.read_csv(train_file, dtype=col_types)
    test_data = pandas.read_csv(test_file, dtype=col_types)
    # print(train_data)
    # print(test_data)
    results = reg_model(train_data, test_data)
    results.to_csv(results_file, index=False)
