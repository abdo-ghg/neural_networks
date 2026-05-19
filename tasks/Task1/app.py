import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SLP_ADALINE as backend


st.set_page_config(page_title="Perceptron & Adaline Classifier", layout="wide")


st.title("Perceptron & Adaline Classifier Visualization")


df = pd.read_csv('penguins.csv')
feature_columns = df.columns.tolist()
feature_columns.remove('Species')  
available_classes = df['Species'].unique().tolist()

feature_options = {col: idx + 1 for idx, col in enumerate(feature_columns)}  # +1 because Species is at index 0


st.header("Model Parameters")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Feature Selection")
    selected_features = st.multiselect(
        "Select Two Features",
        options=feature_columns,
        default=[],
        max_selections=2,
        help="Select exactly two features for training"
    )

    st.subheader("Class Selection")
    selected_classes = st.multiselect(
        "Select Two Classes",
        options=available_classes,
        default=[],
        max_selections=2,
        help="Select exactly two classes for binary classification"
    )

with col_right:
    st.subheader("Hyperparameters")
    learning_rate = st.number_input(
        "Learning Rate (eta)",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f"
    )

    epochs = st.number_input(
        "Number of Epochs (m)",
        min_value=1,
        max_value=10000,
        value=50,
        step=1
    )

    mse_threshold = st.number_input(
        "MSE Threshold",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f"
    )

    add_bias = st.checkbox("Add Bias", value=True)

    algorithm = st.radio(
        "Select Algorithm",
        options=["Perceptron", "Adaline"],
        index=0,
        horizontal=True
    )


st.divider()

valid_input = True
if len(selected_features) != 2:
    st.warning("⚠️ Please select exactly two features.")
    valid_input = False

if len(selected_classes) != 2:
    st.warning("⚠️ Please select exactly two classes.")
    valid_input = False


train_button = st.button("🚀 Train Model", disabled=not valid_input, use_container_width=True)


if train_button and valid_input:
    feature_indices = [feature_options[f] for f in selected_features]
    
    X0 = 1.0 if add_bias else 0.0
    
    with st.spinner("Training model..."):
        try:
            X, y = backend.preprocessing(feature_indices, selected_classes)
            
            X_train, X_test, y_train, y_test = backend.custom_train_test_split(X, y)
            
            if algorithm == "Perceptron":
                w0, w1, w2 = backend.SLP_train(X_train, y_train, epochs, learning_rate, X0)
                accuracy, precision, recall, F1, conf_mat = backend.SLP_test(X_test, y_test, X0, w0, w1, w2)
            else:  
                w0, w1, w2 = backend.ADA_train(X_train, y_train, epochs, learning_rate, X0, mse_threshold)
                accuracy, precision, recall, F1, conf_mat = backend.ADA_test(X_test, y_test, X0, w0, w1, w2)
            
            st.success("✅ Training Complete!")
            

            st.header("After Training")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:

                st.subheader("Decision Boundary Plot")
                
                fig, ax = plt.subplots(figsize=(7, 5))
                
                scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k')
                
                x_vals = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
                y_vals = -(w1/w2) * x_vals - (w0/w2)
                ax.plot(x_vals, y_vals, color='black', label='Decision Boundary')
                
                ax.set_xlabel(selected_features[0], fontsize=12)
                ax.set_ylabel(selected_features[1], fontsize=12)
                ax.set_title(f"{algorithm} Decision Boundary", fontsize=14)
                ax.legend(loc='best')
                
                st.pyplot(fig)
                plt.close()
                
                preds = np.where((w1 * X_test[:, 0] + w2 * X_test[:, 1] + w0) >= 0, 1, -1)
                plot_accuracy = np.mean(preds == y_test)
                st.write(f"**Accuracy calculated from plot side:** {plot_accuracy:.4f}")
            
            with col2:

                st.subheader("Confusion Matrix")
                

                conf_df = pd.DataFrame(
                    conf_mat,
                    columns=[f"Predicted {selected_classes[0]}", f"Predicted {selected_classes[1]}"],
                    index=[f"Actual {selected_classes[0]}", f"Actual {selected_classes[1]}"]
                )
                st.table(conf_df)
                
 
                st.subheader("Performance Metrics")
                
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Value": [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{F1:.4f}"]
                })
                st.table(metrics_df)
                

                st.subheader("Learned Weights")
                st.write(f"**w0 (bias):** {w0:.4f}")
                st.write(f"**w1:** {w1:.4f}")
                st.write(f"**w2:** {w2:.4f}")
            
            st.divider()
            st.subheader("Training Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Training Samples", len(X_train))
            with info_col2:
                st.metric("Testing Samples", len(X_test))
            with info_col3:
                st.metric("Test Accuracy", f"{accuracy:.2%}")
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.exception(e)

else:
    st.info("� Configure the model parameters above and click **Train Model** to begin.")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Features")
        for feat in feature_columns:
            st.write(f"• {feat}")
    
    with col2:
        st.subheader("Available Classes")
        for cls in available_classes:
            count = len(df[df['Species'] == cls])
            st.write(f"• {cls} ({count} samples)")
