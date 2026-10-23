
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "CMAPS_Data"


# NASA C-MAPSS column names
columns = [
    "unit_number",
    "time_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def load_test_data(dataset):
    """Load the selected NASA test file."""
    file_path = DATA_DIR / f"test_{dataset}.txt"

    if not file_path.exists():
        st.error(f"NASA test file not found: {file_path}")
        return None

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
    )


def get_model_features(model):
    """Get the feature names stored by the trained model."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


def main():
    st.set_page_config(
        page_title="NASA C-MAPSS RUL Prediction",
        page_icon="✈️",
        layout="wide",
    )

    st.title("NASA C-MAPSS Remaining Useful Life Prediction")
    st.write(
        "Predict the Remaining Useful Life (RUL) of an aircraft engine "
        "using trained machine learning models."
    )

    dataset = st.selectbox(
        "Select NASA dataset",
        ["FD001", "FD002", "FD003", "FD004"],
    )

    input_mode = st.radio(
        "Choose input mode",
        ["NASA Test Data", "Manual / New Engine Data"],
    )

    model_path = MODEL_DIR / f"{dataset.lower()}_model.pkl"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return

    try:
        model = joblib.load(model_path)
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        return

    expected_features = get_model_features(model)

    if expected_features is None:
        st.error(
            "This model does not expose feature_names_in_. "
            "We need to check the original training code to identify "
            "the exact input features."
        )
        return

    st.caption(f"Model loaded: {dataset}")

    # Display the exact features expected by the saved model
    with st.expander("Check model input features"):
        st.write("Features expected by the model:")
        st.write(expected_features)

    if input_mode == "NASA Test Data":
        test_df = load_test_data(dataset)

        if test_df is None:
            return

        engine_ids = sorted(test_df["unit_number"].unique())

        engine_id = st.selectbox(
            "Select engine number",
            engine_ids,
        )

        engine_data = test_df[
            test_df["unit_number"] == engine_id
        ].sort_values("time_cycles")

        if engine_data.empty:
            st.warning("No data found for this engine.")
            return

        # Check whether all model features exist in the NASA data
        missing_features = [
            feature
            for feature in expected_features
            if feature not in engine_data.columns
        ]

        if missing_features:
            st.error(
                f"NASA data is missing required features: "
                f"{missing_features}"
            )
            return

        # Use the model's original feature names and order
        input_data = engine_data.iloc[[-1]][expected_features]

        st.subheader("Selected engine data")
        st.dataframe(input_data)

        if st.button("Predict RUL"):
            try:
                prediction = model.predict(input_data)[0]

                st.success(
                    f"Predicted Remaining Useful Life: "
                    f"{max(0, prediction):.2f} cycles"
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    else:
        st.subheader("Enter engine sensor values")
        st.info(
            "Enter the values for the features expected by the model."
        )

        values = {}

        for feature in expected_features:
            if feature == "time_cycles":
                values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=1.0,
                    step=1.0,
                )
            elif feature.startswith("operational_setting"):
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.5f",
                )
            else:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.5f",
                )

        if st.button("Predict RUL"):
            input_data = pd.DataFrame(
                [values],
                columns=expected_features,
            )

            try:
                prediction = model.predict(input_data)[0]

                st.success(
                    f"Predicted Remaining Useful Life: "
                    f"{max(0, prediction):.2f} cycles"
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")


if __name__ == "__main__":
    main()
