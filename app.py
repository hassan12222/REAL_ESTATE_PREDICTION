
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Dubai Property Intelligence",
    page_icon="🏙️",
    layout="wide"
)

# -----------------------------
# 2. LOAD SAVED FILES
# -----------------------------
@st.cache_resource
def load_models():
    model_a = joblib.load("model_A_raw_linear_regression.joblib")
    model_b = joblib.load("model_B_log_linear_regression.joblib")
    info = joblib.load("model_metadata.joblib")
    return model_a, model_b, info


@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_modelling_dataset.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["size_sqft"] = pd.to_numeric(df["size_sqft"], errors="coerce")
    df["actual_worth"] = pd.to_numeric(df["actual_worth"], errors="coerce")
    df["price_per_sqft"] = pd.to_numeric(
        df["price_per_sqft"], errors="coerce"
    )
    df["has_parking"] = pd.to_numeric(
        df["has_parking"], errors="coerce"
    )

    df = df.dropna(
        subset=["sale_date", "year", "size_sqft", "actual_worth"]
    )
    return df


try:
    model_a, model_b, info = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Could not load the project files: {e}")
    st.info(
        "Keep app.py, the cleaned CSV, both model files, "
        "and model_metadata.joblib in the same folder."
    )
    st.stop()


# -----------------------------
# 3. HELPER FUNCTIONS
# -----------------------------
def money(value):
    return f"AED {value:,.0f}"


def price_millions(value):
    return f"AED {value / 1_000_000:.2f}M"


def make_features(
    community, property_type, rooms,
    size_sqft, parking, offplan, year
):
    # Must match the exact feature names used during training.
    return pd.DataFrame([{
        "size_sqft": float(size_sqft),
        "log_size": float(np.log(size_sqft)),
        "has_parking": int(parking),
        "offplan": int(offplan),
        "year": int(year),
        "area_name_en": community,
        "property_sub_type_en": property_type,
        "rooms": rooms
    }])[info["features"]]


def evaluate(model, X_test, y_test, log_model=False):
    raw = model.predict(X_test)
    predictions = np.exp(raw) if log_model else raw

    ape = np.abs(y_test.to_numpy() - predictions) / y_test.to_numpy()

    return {
        "R²": r2_score(y_test, predictions),
        "MAE (AED)": mean_absolute_error(y_test, predictions),
        "RMSE (AED)": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "MAPE (%)": ape.mean() * 100,
        "Within 20% (%)": (ape <= 0.20).mean() * 100
    }, predictions


# Reproduce the original 80/20 split for a fair model evaluation.
numbers = [
    "size_sqft", "log_size", "has_parking", "offplan", "year"
]
words = ["area_name_en", "property_sub_type_en", "rooms"]

X = df[numbers + words].copy()
X["has_parking"] = pd.to_numeric(X["has_parking"], errors="coerce")
X = X.dropna(subset=numbers)
y = df.loc[X.index, "actual_worth"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

metrics_a, pred_a = evaluate(model_a, X_test, y_test)
metrics_b, pred_b = evaluate(model_b, X_test, y_test, log_model=True)

# -----------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🏙️ Dubai Property AI")
st.sidebar.caption("Real Estate Intelligence Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Market Overview",
        "💰 Price Predictor",
        "📊 Market Explorer",
        "🧠 Model Insights"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Data: Dubai real estate transactions")
st.sidebar.caption(
    f"Records loaded: {len(df):,}"
)


# =====================================================
# PAGE 1: MARKET OVERVIEW
# =====================================================
if page == "🏠 Market Overview":

    st.title("Dubai Real Estate Market Overview")
    st.write(
        "Explore property sales, community pricing, "
        "and real estate market patterns."
    )

    # Interactive filters
    c1, c2, c3 = st.columns(3)

    years = sorted(df["year"].dropna().astype(int).unique())
    communities = sorted(df["area_name_en"].dropna().unique())
    property_types = sorted(
        df["property_sub_type_en"].dropna().unique()
    )

    with c1:
        selected_years = st.multiselect(
            "Select transaction years",
            years,
            default=years
        )

    with c2:
        selected_types = st.multiselect(
            "Property types",
            property_types,
            default=property_types
        )

    with c3:
        selected_communities = st.multiselect(
            "Communities",
            communities,
            default=communities
        )

    filtered = df[
        df["year"].isin(selected_years)
        & df["property_sub_type_en"].isin(selected_types)
        & df["area_name_en"].isin(selected_communities)
    ]

    if filtered.empty:
        st.warning("No transactions match these filters.")
        st.stop()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Transactions", f"{len(filtered):,}")
    k2.metric("Median Property Price", price_millions(
        filtered["actual_worth"].median()
    ))
    k3.metric("Median Price / Sq Ft", money(
        filtered["price_per_sqft"].median()
    ))
    k4.metric(
        "Communities",
        f"{filtered['area_name_en'].nunique():,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        community_prices = (
            filtered.groupby("area_name_en")["actual_worth"]
            .median()
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            community_prices,
            x="actual_worth",
            y="area_name_en",
            orientation="h",
            title="Median Property Price by Community",
            labels={
                "actual_worth": "Median price (AED)",
                "area_name_en": "Community"
            },
            color="actual_worth",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        yearly = (
            filtered.groupby("year")["price_per_sqft"]
            .median()
            .reset_index()
        )

        fig = px.line(
            yearly,
            x="year",
            y="price_per_sqft",
            markers=True,
            title="Median Price per Sq Ft Over Time",
            labels={
                "year": "Year",
                "price_per_sqft": "AED per sq ft"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transaction Records")
    st.dataframe(
        filtered[
            [
                "sale_date", "area_name_en",
                "property_sub_type_en", "rooms",
                "size_sqft", "actual_worth",
                "price_per_sqft"
            ]
        ].sort_values("sale_date", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download filtered transactions",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dubai_transactions.csv",
        mime="text/csv"
    )


# =====================================================
# PAGE 2: PRICE PREDICTOR
# =====================================================
elif page == "💰 Price Predictor":

    st.title("Property Price Predictor")
    st.write(
        "Enter the property characteristics to estimate "
        "its price using your trained log-price regression model."
    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:
            community = st.selectbox(
                "Community",
                sorted(df["area_name_en"].dropna().unique())
            )

            property_type = st.selectbox(
                "Property type",
                sorted(df["property_sub_type_en"].dropna().unique())
            )

            room_options = sorted(
                df["rooms"].dropna().astype(str).unique()
            )
            preferred = ["Studio", "1 B/R", "2 B/R", "3 B/R",
                         "4 B/R", "5 B/R", "6 B/R", "PENTHOUSE"]
            room_options = (
                [r for r in preferred if r in room_options]
                + [r for r in room_options if r not in preferred]
            )

            rooms = st.selectbox("Rooms", room_options)

        with c2:
            subtype_sizes = info["size_range_by_subtype"].get(
                property_type, {}
            )
            lower = subtype_sizes.get(0.005, 300)
            upper = subtype_sizes.get(0.995, 5000)

            lower = max(100, int(lower))
            upper = max(lower + 1, int(upper))

            size_sqft = st.number_input(
                "Property size (sq ft)",
                min_value=100,
                max_value=100000,
                value=int(np.clip(1000, lower, upper)),
                step=50
            )

            parking_choice = st.selectbox(
                "Has parking?",
                ["Yes", "No"]
            )
            offplan_choice = st.selectbox(
                "Property status",
                ["Ready", "Off-plan"]
            )
            year = st.selectbox(
                "Valuation / reference year",
                list(range(
                    int(df["year"].min()),
                    int(df["year"].max()) + 1
                )),
                index=len(range(
                    int(df["year"].min()),
                    int(df["year"].max()) + 1
                )) - 1
            )

        submitted = st.form_submit_button(
            "✨ Predict Property Price",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        if size_sqft <= 0:
            st.error("Property size must be greater than zero.")
            st.stop()

        features = make_features(
            community=community,
            property_type=property_type,
            rooms=rooms,
            size_sqft=size_sqft,
            parking=1 if parking_choice == "Yes" else 0,
            offplan=1 if offplan_choice == "Off-plan" else 0,
            year=year
        )

        # Model B predicts log(price), so exponentiate.
        prediction = float(np.exp(model_b.predict(features)[0]))

        low = prediction * info["range_low_factor"]
        high = prediction * info["range_high_factor"]

        st.divider()
        st.subheader("Your Property Valuation")

        a, b, c = st.columns(3)
        a.metric("Estimated Price", price_millions(prediction))
        b.metric("Lower Reference", price_millions(low))
        c.metric("Upper Reference", price_millions(high))

        st.caption(
            "The reference range uses historical test-set residuals. "
            "It is a heuristic range, not a guaranteed sale price "
            "or a formal prediction interval."
        )

        st.subheader("Property Summary")
        summary = pd.DataFrame({
            "Property detail": [
                "Community", "Property type", "Rooms",
                "Size (sq ft)", "Parking", "Status",
                "Reference year"
            ],
            "Selected value": [
                community, property_type, rooms,
                f"{size_sqft:,}", parking_choice,
                offplan_choice, year
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.subheader("Estimated Price per Sq Ft")
        st.metric(
            "AED per sq ft",
            money(prediction / size_sqft)
        )

        result = pd.DataFrame([{
            "community": community,
            "property_type": property_type,
            "rooms": rooms,
            "size_sqft": size_sqft,
            "parking": parking_choice,
            "status": offplan_choice,
            "reference_year": year,
            "predicted_price_AED": round(prediction, 2),
            "lower_reference_AED": round(low, 2),
            "upper_reference_AED": round(high, 2)
        }])

        st.download_button(
            "⬇️ Download valuation report (CSV)",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="property_valuation.csv",
            mime="text/csv"
        )


# =====================================================
# PAGE 3: MARKET EXPLORER
# =====================================================
elif page == "📊 Market Explorer":

    st.title("Interactive Market Explorer")
    st.write("Compare prices across different property segments.")

    chart_choice = st.selectbox(
        "Choose an analysis",
        [
            "Community comparison",
            "Property type comparison",
            "Room type comparison",
            "Off-plan vs ready",
            "Property size vs price",
            "Price distribution"
        ]
    )

    if chart_choice == "Community comparison":
        metric = st.radio(
            "Price metric",
            ["Median property price", "Median price per sq ft"],
            horizontal=True
        )

        column = (
            "actual_worth"
            if metric == "Median property price"
            else "price_per_sqft"
        )

        chart_data = (
            df.groupby("area_name_en")[column]
            .median()
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            chart_data,
            x=column,
            y="area_name_en",
            orientation="h",
            title=metric + " by Community",
            color=column,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Property type comparison":
        chart_data = (
            df.groupby("property_sub_type_en")["actual_worth"]
            .median()
            .reset_index()
        )
        fig = px.bar(
            chart_data,
            x="property_sub_type_en",
            y="actual_worth",
            color="property_sub_type_en",
            title="Median Price by Property Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Room type comparison":
        chart_data = (
            df.groupby("rooms")["actual_worth"]
            .median()
            .sort_values()
            .reset_index()
        )
        fig = px.bar(
            chart_data,
            x="rooms",
            y="actual_worth",
            title="Median Price by Room Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Off-plan vs ready":
        chart_data = (
            df.groupby(["year", "offplan"])["price_per_sqft"]
            .median()
            .reset_index()
        )
        chart_data["Status"] = chart_data["offplan"].map({
            0: "Ready",
            1: "Off-plan"
        })
        fig = px.line(
            chart_data,
            x="year",
            y="price_per_sqft",
            color="Status",
            markers=True,
            title="Off-plan vs Ready Price per Sq Ft"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Property size vs price":
        sample = df.sample(
            min(10000, len(df)), random_state=42
        )
        fig = px.scatter(
            sample,
            x="size_sqft",
            y="actual_worth",
            color="property_sub_type_en",
            hover_data=["area_name_en", "rooms"],
            log_x=True,
            log_y=True,
            title="Property Size vs Sale Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Price distribution":
        fig = px.histogram(
            df,
            x="actual_worth",
            nbins=60,
            title="Distribution of Property Sale Prices",
            labels={"actual_worth": "Sale price (AED)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Community Ranking Table")

    ranking = (
        df.groupby("area_name_en")
        .agg(
            Sales=("actual_worth", "size"),
            Median_Price=("actual_worth", "median"),
            Median_Price_Per_Sq_Ft=("price_per_sqft", "median")
        )
        .sort_values("Median_Price", ascending=False)
        .reset_index()
    )

    st.dataframe(
        ranking.style.format({
            "Median_Price": "AED {:,.0f}",
            "Median_Price_Per_Sq_Ft": "AED {:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download community rankings",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name="community_rankings.csv",
        mime="text/csv"
    )


# =====================================================
# PAGE 4: MODEL INSIGHTS
# =====================================================
elif page == "🧠 Model Insights":

    st.title("Machine Learning Model Insights")
    st.write(
        "Evaluate both saved Linear Regression models on the "
        "same held-out test data."
    )

    st.info(
        "Model A predicts the original price. Model B predicts "
        "log(price) and converts its predictions back to AED."
    )

    comparison = pd.DataFrame({
        "Model A — Raw Price": metrics_a,
        "Model B — Log Price": metrics_b
    }).T

    st.subheader("Evaluation Metrics")

    display_metrics = comparison.copy()
    display_metrics["R²"] = display_metrics["R²"].map(
        lambda x: f"{x:.4f}"
    )
    for col in ["MAE (AED)", "RMSE (AED)"]:
        display_metrics[col] = display_metrics[col].map(
            lambda x: f"AED {x:,.0f}"
        )
    for col in ["MAPE (%)", "Within 20% (%)"]:
        display_metrics[col] = display_metrics[col].map(
            lambda x: f"{x:.2f}%"
        )

    st.dataframe(display_metrics, use_container_width=True)

    st.caption(
        "R² is better when higher; MAE, RMSE and MAPE are generally "
        "better when lower. The within-20% metric is better when higher."
    )

    selected_model = st.radio(
        "Inspect predictions from",
        ["Model A — Raw Price", "Model B — Log Price"],
        horizontal=True
    )

    predictions = pred_a if selected_model.startswith("Model A") else pred_b

    results = X_test.copy()
    results["actual"] = y_test.to_numpy()
    results["predicted"] = predictions
    results["absolute_error"] = abs(
        results["actual"] - results["predicted"]
    )
    results["error_percent"] = (
        results["absolute_error"] / results["actual"] * 100
    )

    fig = px.scatter(
        results,
        x="actual",
        y="predicted",
        color="property_sub_type_en",
        hover_data=["area_name_en", "rooms", "size_sqft"],
        log_x=True,
        log_y=True,
        title="Actual vs Predicted Property Prices",
        labels={
            "actual": "Actual price (AED)",
            "predicted": "Predicted price (AED)"
        }
    )

    # Ideal predictions lie on y = x.
    low = min(results["actual"].min(), results["predicted"].min())
    high = max(results["actual"].max(), results["predicted"].max())

    fig.add_shape(
        type="line",
        x0=low, y0=low,
        x1=high, y1=high,
        line=dict(dash="dash", color="red")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Error Analysis")

    err1, err2 = st.columns(2)

    with err1:
        fig = px.histogram(
            results,
            x="error_percent",
            nbins=50,
            title="Absolute Percentage Error",
            labels={"error_percent": "Absolute error (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with err2:
        by_type = (
            results.groupby("property_sub_type_en")["error_percent"]
            .mean()
            .reset_index()
            .sort_values("error_percent")
        )
        fig = px.bar(
            by_type,
            x="property_sub_type_en",
            y="error_percent",
            title="Mean Absolute Percentage Error by Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Test Predictions")
    st.dataframe(
        results[
            [
                "area_name_en", "property_sub_type_en", "rooms",
                "size_sqft", "actual", "predicted",
                "absolute_error", "error_percent"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download test predictions",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="model_test_predictions.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("Dubai Property Intelligence | ML Demo")

