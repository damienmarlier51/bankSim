from lifetimes import GammaGammaFitter, \
                      BetaGeoFitter
from pathlib import Path
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
raw_data_dir = "{}/{}".format(project_dir, "data/raw")
output_data_dir = "{}/{}".format(project_dir, "data/output")


if __name__ == "__main__":

    dataset_filepath = "{}/{}".format(raw_data_dir, "bs140513_032310.csv")
    df = pd.read_csv(dataset_filepath)

    T = df.groupby("customer")[["step"]].max()
    recency = T - df.groupby("customer")[["step"]].min()
    monetary = df.groupby(["customer", "step"])[["amount"]].mean() \
                                                           .reset_index() \
                                                           .groupby("customer")[["amount"]] \
                                                           .mean()
    frequency = df.drop_duplicates(subset=["customer", "step"],
                                   keep="first").groupby(["customer"]) \
                                                .count() - 1

    recency.rename(columns={"step": "recency"}, inplace=True)
    frequency.rename(columns={"step": "frequency"}, inplace=True)
    T.rename(columns={"step": "T"}, inplace=True)
    monetary.rename(columns={"amount": "monetary_value"}, inplace=True)

    df_rfm = pd.concat([recency, T, monetary, frequency], axis=1)
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(frequency=df_rfm["frequency"],
            monetary_value=df_rfm["monetary_value"])

    df_rfm["expected_monetary_value"] = df_rfm.apply(lambda row: ggf.conditional_expected_average_profit(
                                                                        row["frequency"],
                                                                        row["monetary_value"]
                                                                        ), axis=1)

    bgf = BetaGeoFitter(penalizer_coef=1)
    bgf.fit(frequency=df_rfm["frequency"],
            recency=df_rfm["recency"],
            T=df_rfm["T"])

    df_rfm["pred_nb_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(t=180,
                                                                                          frequency=df_rfm["frequency"],
                                                                                          recency=df_rfm["recency"],
                                                                                          T=df_rfm["T"])

    df_rfm["pred_revenue"] = df_rfm.apply(lambda row: row["pred_nb_purchases"] * row["expected_monetary_value"], axis=1)

    df_rfm.sort_values(by="pred_revenue", inplace=True)
    df_rfm.to_csv("{}/clv.csv".format(output_data_dir))
