import numpy as np  # noqa
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn import cluster
from sklearn.linear_model import LinearRegression


def hist_generator(data, ax):
    def wrap(col):
        return data[col].hist(ax=ax, bins=10, histtype="step")
        return pd.to_numeric(data[col]).hist(ax=ax, bins=10, histtype="step")

    return wrap


def overview(data):
    st.write(f"Total {len(data)} entries")
    st.markdown("## Column descriptions")
    for column in data.columns:
        st.markdown(f"### {column}\n```{data.dtypes[column]}```")
        if data.dtypes[column] == object:
            st.write(f"{data[column].nunique()} unique values")
        else:
            st.write(f"min: {data[column].min()}")
            st.write(f"median: {data[column].median()}")
            st.write(f"max: {data[column].max()}")

        if not data.dtypes[column] == object or data[column].nunique() < 1000:
            st.write("Histogram")
            fig, ax = plt.subplots(1, 1)
            ax.locator_params(axis="x", nbins=7)
            hist_generator(data, ax)(column)
            st.pyplot(fig)


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
            # cluster.AffinityPropagation,
            cluster.KMeans,
            cluster.MiniBatchKMeans,
            cluster.AgglomerativeClustering,
            cluster.Birch,
            # cluster.DBSCAN,
            # cluster.FeatureAgglomeration,
            # cluster.MeanShift,
            # cluster.OPTICS,
            # cluster.SpectralClustering,
            # cluster.SpectralBiclustering,
            # cluster.SpectralCoclustering,
        ]
    }

    selectedmethodname = st.sidebar.selectbox(
        "Clustering method",
        list(CLUSTERING_METHODS.keys()),
    )
    selectedmethod = CLUSTERING_METHODS[selectedmethodname]
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
    clustered = selectedmethod(n_clusters=n_clusters).fit(normalized)
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("hsv", n_clusters)
    ax.scatter(x=normalized[dim1], y=normalized[dim2], c=clustered.labels_, cmap=cmap)
    ax.set_xlabel(dim1)
    ax.set_ylabel(dim2)
    logscale = st.sidebar.checkbox("Log-scale of data")
    if logscale:
        ax.set_yscale("log")
        ax.set_xscale("log")
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
    st.write("Input data set (first 100 entries)")
    st.write(data[:100])

    VIEWS = {
        "Data overview": overview,
        "Regression": regression,
        "Coorelation": coorelation,
        "Clustering": clustering,
    }
    view = st.sidebar.radio("View", list(VIEWS.keys()))
    st.sidebar.markdown("---")
    VIEWS.get(view)(data)


main()
