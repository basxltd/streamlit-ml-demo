import altair as alt
import numpy as np  # noqa
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn import cluster
from sklearn.linear_model import LinearRegression


def alldata(data):
    st.write(data[:100])


def profit(data, agg_func, typelabel):
    st.markdown(f"## {typelabel} numbers")

    def countries():
        st.markdown("Summed up profits by country")
        countrydata = (
            data.loc[:, ["Profit", "Country"]]
            .groupby("Country")
            .agg(agg_func)
            .reset_index()
            .sort_values("Profit")
        )
        top, bottom = st.beta_columns(2)
        top.markdown("### Top countries")
        top.altair_chart(
            alt.Chart(countrydata[-10:])
            .mark_bar()
            .encode(
                x=alt.X("Country", sort="-y"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )
        bottom.markdown("### Bottom countries")
        bottom.altair_chart(
            alt.Chart(countrydata[:10])
            .mark_bar()
            .encode(
                x=alt.X("Country", sort="-y"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

    def market():
        st.markdown("Summed up profits by market")
        worlddata = (
            data.loc[:, ["Profit", "Market"]]
            .groupby("Market")
            .agg(agg_func)
            .reset_index()
            .sort_values("Profit")
        )
        st.altair_chart(
            alt.Chart(worlddata)
            .mark_bar()
            .encode(
                x=alt.X("Market", sort="-y"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

    def weekdays():
        st.markdown("Summed up profits by day of the week")

        weekdaydata = data.loc[:, ["Profit", "Order Date"]].copy()
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekdaydata["Weekday"] = [
            f"{days.index(i)} {i}" for i in weekdaydata["Order Date"].dt.day_name()
        ]
        weekdaydata = (
            weekdaydata.groupby("Weekday")
            .agg(agg_func)
            .reset_index()
            .sort_values("Weekday")
        )

        st.altair_chart(
            alt.Chart(weekdaydata)
            .mark_bar()
            .encode(
                x=alt.X("Weekday", sort="x"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

    def months():
        st.markdown("Summed up profits by month")

        monthdata = data.loc[:, ["Profit", "Order Date"]].copy()
        days = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        monthdata["Month"] = [
            f"{days.index(i):02} {i}" for i in monthdata["Order Date"].dt.month_name()
        ]
        monthdata = (
            monthdata.groupby("Month").agg(agg_func).reset_index().sort_values("Month")
        )

        st.altair_chart(
            alt.Chart(monthdata)
            .mark_bar()
            .encode(
                x=alt.X("Month", sort="x"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

    def producttype():
        st.markdown("Summed up profits by product type")
        allproducttypedata = data.loc[:, ["Profit", "Category", "Sub-Category"]]
        producttypedata = (
            allproducttypedata.groupby("Category")
            .agg(agg_func)
            .reset_index()
            .sort_values("Profit")
        )
        st.altair_chart(
            alt.Chart(producttypedata)
            .mark_bar()
            .encode(
                x=alt.X("Category", sort="-y"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

        st.markdown("Summed up profits by product sub-type")
        subproducttypedata = (
            allproducttypedata.groupby("Sub-Category")
            .agg(agg_func)
            .reset_index()
            .sort_values("Profit")
        )
        st.altair_chart(
            alt.Chart(subproducttypedata)
            .mark_bar()
            .encode(
                x=alt.X("Sub-Category", sort="-y"),
                y=alt.Y("Profit", axis=alt.Axis(title=typelabel)),
            ),
            use_container_width=True,
        )

    TYPES = {
        "Countries": countries,
        "Market": market,
        "Day of the week": weekdays,
        "Month": months,
        "Type of product": producttype,
    }
    datatype = st.radio("Choose dimension", list(TYPES.keys()))
    st.markdown("---")
    TYPES[datatype]()


def hist_generator(data, ax):
    def wrap(col):
        return data[col].hist(ax=ax, bins=10, histtype="step")
        return pd.to_numeric(data[col]).hist(ax=ax, bins=10, histtype="step")

    return wrap


def overview(data):
    st.write(f"Total {len(data)} entries")
    col1, col2 = st.beta_columns([1, 2])
    selected_column = col1.radio("Choose column", list(data.columns))

    col2.markdown(f"### {selected_column}\n```{data.dtypes[selected_column]}```")
    if data.dtypes[selected_column] == object:
        col2.write(f"{data[selected_column].nunique()} unique values")
    else:
        col2.write(f"min: {data[selected_column].min()}")
        col2.write(f"median: {data[selected_column].median()}")
        col2.write(f"max: {data[selected_column].max()}")

    if (
        not data.dtypes[selected_column] == object
        or data[selected_column].nunique() < 1000
    ):
        col2.write("Histogram")
        fig, ax = plt.subplots(1, 1)
        ax.locator_params(axis="x", nbins=7)
        hist_generator(data, ax)(selected_column)
        col2.pyplot(fig)


def regression(data):
    allowed_columns = []
    for column in data.columns:
        if is_numeric_dtype(data.dtypes[column]):
            allowed_columns.append(column)
    independent = st.sidebar.selectbox(
        "Input value (independent variable)",
        allowed_columns,
    )
    grouping = st.sidebar.checkbox("Group input value", value=True)
    allowed_columns2 = [i for i in allowed_columns if i != independent]
    dependent = st.sidebar.selectbox(
        "Value to prediction (dependent variable)",
        allowed_columns2,
    )

    if grouping:
        AGGREGATIONS = {"sum": lambda a: a.sum(), "mean": lambda a: a.mean()}
        aggregation = st.sidebar.selectbox("Aggregation", list(AGGREGATIONS.keys()))

        learndata = AGGREGATIONS[aggregation](data.groupby(independent)).loc[
            :, [dependent]
        ]
        X_labeled = pd.DataFrame(learndata.index)[independent]
    else:
        learndata = data.loc[:, [dependent]]
        X_labeled = data.loc[:, [independent]]

    if independent == "Date":
        X = pd.to_datetime(X_labeled).dt.strftime("%m%d%Y").astype(int)
    else:
        X = X_labeled

    # learning
    X = X.to_numpy().reshape(-1, 1)
    y = learndata[dependent]
    linreg = LinearRegression()
    linreg.fit(X, y)

    # plotting
    fig, ax = plt.subplots()
    ax.plot(X_labeled, y, color="g", label="Historical Data")
    ax.plot(X_labeled, linreg.predict(X), color="k", label="Regression Curve")
    ax.set_ylabel(dependent)
    ax.set_xlabel(independent)
    ax.set_title("Regression")
    st.pyplot(fig)


def clustering(data):
    allowed_columns = []
    for column in data.columns:
        if is_numeric_dtype(data.dtypes[column]):
            allowed_columns.append(column)
    dim1 = st.sidebar.selectbox("Dimension 1", allowed_columns, index=2)
    allowed_columns2 = [i for i in allowed_columns if i != dim1]
    dim2 = st.sidebar.selectbox("Dimension 2", allowed_columns2, index=2)
    CLUSTERING_METHODS = {
        algo.__name__: algo
        for algo in [
            cluster.KMeans,
            cluster.MiniBatchKMeans,
            cluster.AgglomerativeClustering,
            cluster.Birch,
            cluster.DBSCAN,
        ]
    }

    selectedmethodname = st.sidebar.selectbox(
        "Clustering method",
        list(CLUSTERING_METHODS.keys()),
    )
    selectedmethod = CLUSTERING_METHODS[selectedmethodname]
    eps = 0.5
    if selectedmethodname == "DBSCAN":
        eps = st.sidebar.number_input(label="Density", value=0.5)
    else:
        n_clusters = st.sidebar.number_input(label="Number of clusters", value=8)
    datapoints = st.sidebar.number_input(label="Number of datapoints", value=1000)

    # plotting
    normalized = data.loc[:datapoints, [dim1, dim2]].copy()

    normalize = st.sidebar.checkbox("Normalize data")
    if normalize:
        for column in normalized.columns:
            normalized[column] = (normalized[column] - normalized[column].min()) / (
                normalized[column].max() - normalized[column].min()
            )

    normalized = normalized.fillna(0)
    if selectedmethodname == "DBSCAN":
        clustered = selectedmethod(eps).fit(normalized)
        n_clusters = len(set(clustered.labels_))
    else:
        clustered = selectedmethod(n_clusters=n_clusters).fit(normalized)
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("hsv", n_clusters)
    colors = np.ma.array(clustered.labels_, mask=clustered.labels_ < 0)
    ax.scatter(x=normalized[dim1], y=normalized[dim2], c=colors, cmap=cmap)
    ax.set_xlabel(dim1)
    ax.set_ylabel(dim2)
    logscale = st.sidebar.checkbox("Log-scale of data")
    if logscale:
        ax.set_yscale("log")
        ax.set_xscale("log")
    ax.axis("square")
    ax.set_title("Clustering")
    st.pyplot(fig)


@st.cache
def descriptions():
    with open("data.txt") as f:
        return f.read()


def coorelation(data):
    st.markdown("# Correlation of input data")
    st.markdown("*" + descriptions() + "*")
    st.markdown("*(Correlation can only be calcualted on numerical data)*")
    st.write(data.corr())


@st.cache
def association_data(data):
    return list()


@st.cache
def loaddata():
    return pd.read_csv("superstore_sales.csv").astype(
        {"Order Date": "datetime64", "Ship Date": "datetime64"}
    )


def main():
    st.title("Supermarket Sales Analytics")

    data = loaddata()

    view_type = st.sidebar.selectbox(
        "Analytics type", ["Business Data", "Data Exploration"]
    )
    if view_type == "Business Data":
        BUSINESS_VIEWS = {
            "Profit numbers": lambda d: profit(d, sum, "Profit"),
            "Sale numbers": lambda d: profit(d, len, "Sales"),
            "Data": alldata,
        }

        st.sidebar.markdown("## Business analytics")
        view = st.sidebar.radio("Choose a tool", list(BUSINESS_VIEWS.keys()))
        st.sidebar.markdown("---")
        BUSINESS_VIEWS.get(view)(data)
    else:
        st.write("Input data set (first 100 entries)")
        st.write(data[:100])
        ADVANCED_VIEWS = {
            "Data overview": overview,
            "Regression": regression,
            "Coorelation": coorelation,
            "Clustering": clustering,
        }
        st.sidebar.markdown("## Advanced data exploration")
        view = st.sidebar.radio("Choose a tool", list(ADVANCED_VIEWS.keys()))
        st.sidebar.markdown("---")
        ADVANCED_VIEWS.get(view)(data)


main()
