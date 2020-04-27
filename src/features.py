from pathlib import Path
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
raw_data_dir = "{}/{}".format(project_dir, "data/raw")
processed_data_dir = "{}/{}".format(project_dir, "data/processed")


if __name__ == "__main__":

    dataset_filepath = "{}/{}".format(raw_data_dir, "bs140513_032310.csv")
    df = pd.read_csv(dataset_filepath)
    df_dataset = df.loc[df["gender"].isin(["'F'", "'M'"]), :]
    df_dataset.loc[:, "count"] = 1

    merchants = df_dataset["merchant"].unique().tolist()
    categories = df_dataset["category"].unique().tolist()
    df_dataset.sort_values(by="customer", inplace=True)

    # Number of purchases per customer
    df_count = df_dataset.groupby(["customer"]).count()[["count"]]
    # Target variable
    df_gender = df_dataset.drop_duplicates(["customer", "gender"], keep="first").set_index("customer")[["gender"]]

    df_age = df_dataset.drop_duplicates(["customer", "age"], keep="first").set_index("customer")[["age"]]
    df_age = pd.get_dummies(df_age, prefix=['age'])

    # Amount of purchases per customer
    df_amount = df_dataset.groupby(["customer"]).sum()[["amount"]]
    df_mean_amount = df_dataset.groupby(["customer"]).mean()[["amount"]]
    df_mean_amount.rename(columns={"amount": "mean_amount"}, inplace=True)
    df_std_amount = df_dataset.groupby(["customer"]).std()[["amount"]]
    df_std_amount.rename(columns={"amount": "std_amount"}, inplace=True)

    # Number of Categories customer purchased from
    df_count_category = df_dataset.drop_duplicates(["customer", "category"], keep="first").groupby(["customer"]).count()[["count"]]
    df_count_category.rename(columns={"count": "count_category"}, inplace=True)

    # Number of Merchants customer purchased from
    df_count_merchant = df_dataset.drop_duplicates(["customer", "merchant"], keep="first").groupby(["customer"]).count()[["count"]]
    df_count_merchant.rename(columns={"count": "count_merchant"}, inplace=True)

    # Number of purchases per customer and category
    df_count_per_category = df_dataset.groupby(["customer", "category"]).count().reset_index()[["customer", "category", "count"]]
    df_count_per_category = pd.pivot_table(df_count_per_category, values=["count"], index=["customer"], columns=["category"])[["count"]]
    df_count_per_category.columns = list(map("_".join, df_count_per_category.columns))
    df_count_per_category.fillna(value=0, inplace=True)

    # Amount of purchases per customer and category
    df_amount_per_category = df_dataset.groupby(["customer", "category"]).sum().reset_index()[["customer", "category", "amount"]]
    df_amount_per_category = pd.pivot_table(df_amount_per_category, values=["amount"], index=["customer"], columns=["category"])[["amount"]]
    df_amount_per_category.columns = list(map("_".join, df_amount_per_category.columns))
    df_amount_per_category.fillna(value=0, inplace=True)

    df_mean_amount_per_category = df_dataset.groupby(["customer", "category"]).mean().reset_index()[["customer", "category", "amount"]]
    df_mean_amount_per_category = pd.pivot_table(df_mean_amount_per_category, values=["amount"], index=["customer"], columns=["category"])[["amount"]]
    df_mean_amount_per_category.rename(columns={"amount": "mean_amount"}, inplace=True)
    df_mean_amount_per_category.columns = list(map("_".join, df_mean_amount_per_category.columns))
    df_mean_amount_per_category.fillna(value=0, inplace=True)

    df_std_amount_per_category = df_dataset.groupby(["customer", "category"]).std().reset_index()[["customer", "category", "amount"]]
    df_std_amount_per_category = pd.pivot_table(df_std_amount_per_category, values=["amount"], index=["customer"], columns=["category"])[["amount"]]
    df_std_amount_per_category.rename(columns={"amount": "std_amount"}, inplace=True)
    df_std_amount_per_category.columns = list(map("_".join, df_std_amount_per_category.columns))
    df_std_amount_per_category.fillna(value=0, inplace=True)

    # Number of purchases per customer and merchant
    df_count_per_merchant = df_dataset.groupby(["customer", "merchant"]).count().reset_index()[["customer", "merchant", "count"]]
    df_count_per_merchant = pd.pivot_table(df_count_per_merchant, values=["count"], index=["customer"], columns=["merchant"])[["count"]]
    df_count_per_merchant.columns = list(map("_".join, df_count_per_merchant.columns))
    df_count_per_merchant.fillna(value=0, inplace=True)

    # Amount of purchases per customer and merchant
    df_amount_per_merchant = df_dataset.groupby(["customer", "merchant"]).sum().reset_index()[["customer", "merchant", "amount"]]
    df_amount_per_merchant = pd.pivot_table(df_amount_per_merchant, values=["amount"], index=["customer"], columns=["merchant"])[["amount"]]
    df_amount_per_merchant.columns = list(map("_".join, df_amount_per_merchant.columns))
    df_amount_per_merchant.fillna(value=0, inplace=True)

    df_mean_amount_per_merchant = df_dataset.groupby(["customer", "merchant"]).mean().reset_index()[["customer", "merchant", "amount"]]
    df_mean_amount_per_merchant = pd.pivot_table(df_mean_amount_per_merchant, values=["amount"], index=["customer"], columns=["merchant"])[["amount"]]
    df_mean_amount_per_merchant.rename(columns={"amount": "mean_amount"}, inplace=True)
    df_mean_amount_per_merchant.columns = list(map("_".join, df_mean_amount_per_merchant.columns))
    df_mean_amount_per_merchant.fillna(value=0, inplace=True)

    df_std_amount_per_merchant = df_dataset.groupby(["customer", "merchant"]).std().reset_index()[["customer", "merchant", "amount"]]
    df_std_amount_per_merchant = pd.pivot_table(df_std_amount_per_merchant, values=["amount"], index=["customer"], columns=["merchant"])[["amount"]]
    df_std_amount_per_merchant.rename(columns={"amount": "std_amount"}, inplace=True)
    df_std_amount_per_merchant.columns = list(map("_".join, df_std_amount_per_merchant.columns))
    df_std_amount_per_merchant.fillna(value=0, inplace=True)

    # Mean and Std for customer interpurchase time
    df_customer_steps = df_dataset.drop_duplicates(["customer", "step"], keep="first")[["customer", "step"]].sort_values(by=["customer", "step"])
    df_customer_steps["diff"] = df_customer_steps.groupby("customer")["step"].diff()
    df_customer_steps.dropna(subset=["diff"], axis=0, inplace=True)

    # Mean IPT
    df_mean_ipt = df_customer_steps.groupby(["customer"]).mean()[["diff"]]
    df_mean_ipt.rename(columns={"diff": "mean_ipt"}, inplace=True)
    df_mean_ipt.fillna(value=180, inplace=True)

    # Std IPT
    df_std_ipt = df_customer_steps.groupby(["customer"]).std()[["diff"]]
    df_std_ipt.rename(columns={"diff": "std_ipt"}, inplace=True)
    df_std_ipt.fillna(value=0, inplace=True)

    # Mean and Std for customer category interpurchase time
    df_customer_category_steps = df_dataset.drop_duplicates(["customer", "step", "category"], keep="first")[["customer", "step", "category"]].sort_values(by=["customer", "step", "category"])
    df_customer_category_steps["diff"] = df_customer_category_steps.groupby(["customer", "category"])["step"].diff()
    df_customer_category_steps.dropna(subset=["diff"], axis=0, inplace=True)

    df_mean_category_ipt = df_customer_category_steps.groupby(["customer", "category"]).mean().reset_index()[["customer", "category", "diff"]]
    df_mean_category_ipt.rename(columns={"diff": "mean_ipt"}, inplace=True)
    df_mean_category_ipt = pd.pivot_table(df_mean_category_ipt, values=["mean_ipt"], index=["customer"], columns=["category"])
    df_mean_category_ipt.columns = list(map("_".join, df_mean_category_ipt.columns))
    df_mean_category_ipt.fillna(value=180, inplace=True)

    df_std_category_ipt = df_customer_category_steps.groupby(["customer", "category"]).std().reset_index()[["customer", "category", "diff"]]
    df_std_category_ipt.rename(columns={"diff": "std_ipt"}, inplace=True)
    df_std_category_ipt = pd.pivot_table(df_std_category_ipt, values=["std_ipt"], index=["customer"], columns=["category"])
    df_std_category_ipt.columns = list(map("_".join, df_std_category_ipt.columns))
    df_std_category_ipt.fillna(value=0, inplace=True)

    # Mean and Std for customer category interpurchase time
    df_customer_merchant_steps = df_dataset.drop_duplicates(["customer", "step", "merchant"], keep="first")[["customer", "step", "merchant"]].sort_values(by=["customer", "step", "merchant"])
    df_customer_merchant_steps["diff"] = df_customer_merchant_steps.groupby(["customer", "merchant"])["step"].diff()
    df_customer_merchant_steps.dropna(subset=["diff"], axis=0, inplace=True)

    df_mean_merchant_ipt = df_customer_merchant_steps.groupby(["customer", "merchant"]).mean().reset_index()[["customer", "merchant", "diff"]]
    df_mean_merchant_ipt.rename(columns={"diff": "mean_ipt"}, inplace=True)
    df_mean_merchant_ipt = pd.pivot_table(df_mean_merchant_ipt, values=["mean_ipt"], index=["customer"], columns=["merchant"])
    df_mean_merchant_ipt.columns = list(map("_".join, df_mean_merchant_ipt.columns))
    df_mean_merchant_ipt.fillna(value=180, inplace=True)

    df_std_merchant_ipt = df_customer_merchant_steps.groupby(["customer", "merchant"]).std().reset_index()[["customer", "merchant", "diff"]]
    df_std_merchant_ipt.rename(columns={"diff": "std_ipt"}, inplace=True)
    df_std_merchant_ipt = pd.pivot_table(df_std_merchant_ipt, values=["std_ipt"], index=["customer"], columns=["merchant"])
    df_std_merchant_ipt.columns = list(map("_".join, df_std_merchant_ipt.columns))
    df_std_merchant_ipt.fillna(value=0, inplace=True)

    dfs = [df_gender,
           df_age,
           df_count,
           df_amount,
           df_mean_amount,
           df_std_amount,
           df_mean_ipt,
           df_std_ipt,
           df_count_category,
           df_count_per_category,
           df_amount_per_category,
           df_mean_amount_per_category,
           df_std_amount_per_category,
           df_mean_category_ipt,
           df_std_category_ipt,
           df_count_merchant,
           df_count_per_merchant,
           df_amount_per_merchant,
           df_mean_amount_per_merchant,
           df_std_amount_per_merchant,
           df_mean_merchant_ipt,
           df_std_merchant_ipt]

    # Concatenate all features
    df_features = pd.concat(dfs, axis=1, sort=True)

    count_category_columns = ["count_{}".format(category) for category in categories]
    amount_category_columns = ["amount_{}".format(category) for category in categories]
    count_merchant_columns = ["count_{}".format(merchant) for merchant in merchants]
    amount_merchant_columns = ["amount_{}".format(merchant) for merchant in merchants]

    mean_ipt_category_columns = ["mean_ipt_{}".format(category) for category in categories]
    std_ipt_category_columns = ["std_ipt_{}".format(category) for category in categories]

    mean_ipt_merchant_columns = ["mean_ipt_{}".format(merchant) for merchant in merchants]
    std_ipt_merchant_columns = ["std_ipt_{}".format(merchant) for merchant in merchants]

    std_amount_merchant_columns = ["std_amount_{}".format(merchant) for merchant in merchants]

    normalized_count_category_columns = ["ncount_{}".format(category) for category in categories]
    normalized_amount_category_columns = ["namount_{}".format(category) for category in categories]
    normalized_count_merchant_columns = ["ncount_{}".format(merchant) for merchant in merchants]
    normalized_amount_merchant_columns = ["namount_{}".format(merchant) for merchant in merchants]

    # Filling missing values
    df_features[[column for column in df_features.columns.tolist() if column in mean_ipt_category_columns]] = df_features[[column for column in df_features.columns.tolist() if column in mean_ipt_category_columns]].fillna(180)
    df_features[[column for column in df_features.columns.tolist() if column in mean_ipt_merchant_columns]] = df_features[[column for column in df_features.columns.tolist() if column in mean_ipt_merchant_columns]].fillna(180)
    df_features[[column for column in df_features.columns.tolist() if column in std_ipt_category_columns]] = df_features[[column for column in df_features.columns.tolist() if column in std_ipt_category_columns]].fillna(0)
    df_features[[column for column in df_features.columns.tolist() if column in std_ipt_merchant_columns]] = df_features[[column for column in df_features.columns.tolist() if column in std_ipt_merchant_columns]].fillna(0)
    df_features[[column for column in df_features.columns.tolist() if column in std_amount_merchant_columns]] = df_features[[column for column in df_features.columns.tolist() if column in std_amount_merchant_columns]].fillna(0)

    df_features[normalized_count_category_columns] = df_features[count_category_columns].div(df_features["count"], axis=0)
    df_features[normalized_amount_category_columns] = df_features[amount_category_columns].div(df_features["amount"], axis=0)
    df_features[normalized_count_merchant_columns] = df_features[count_merchant_columns].div(df_features["count"], axis=0)
    df_features[normalized_amount_merchant_columns] = df_features[amount_merchant_columns].div(df_features["amount"], axis=0)

    df_features.drop(columns=count_category_columns, inplace=True)
    df_features.drop(columns=amount_category_columns, inplace=True)
    df_features.drop(columns=count_merchant_columns, inplace=True)
    df_features.drop(columns=amount_merchant_columns, inplace=True)

    df_features.replace({"'M'": 0, "'F'": 1}, inplace=True)
    feature_filepath = "{}/features.csv".format(processed_data_dir)
    df_features.to_csv(feature_filepath)
