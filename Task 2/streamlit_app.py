import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from micrograd.engine import Tensor
from model import MLP
import micrograd.optim as optim
from main import preprocessing, split

EPOCHS = 1000
LEARNING_RATE = 0.01
BIAS = True

FEATURES = 5
CLASSES = 3
TANH_HIDDEN = True

NUM_HIDDEN_NEURONS = 64
NUM_OF_HIDDEN_LAYERS = 2


def setup_page():
    st.set_page_config(
        page_title="MLP Backpropagation Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(120deg, #0f1021 0%, #1a1b47 35%, #253b80 65%, #0c8f8f 100%);
                color: #f8f9fa;
            }
            .main-card {
                background: rgba(13, 18, 50, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
                padding: 1.2rem;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.30);
                margin-bottom: 1rem;
            }
            .stMetric {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 0.8rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .header-title {
                font-size: 2.3rem;
                font-weight: 800;
                color: #ffffff;
                margin-bottom: 0.2rem;
            }
            .header-sub {
                font-size: 1.05rem;
                color: #d0d9ff;
                margin-bottom: 1.2rem;
            }
            .stButton > button {
                border-radius: 14px;
                border: none;
                background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcB77, #4d96ff);
                color: #10122b;
                font-weight: 800;
                padding: 0.6rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="header-title">🧠✨ MLP Backpropagation Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">Train, evaluate, and compare your Multi-Layer Perceptron runs with a colorful interactive UI.</div>',
        unsafe_allow_html=True,
    )


def initialize_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def sidebar_config():
    with st.sidebar:
        st.header("⚙️ Network Configuration")

        hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=6, value=NUM_OF_HIDDEN_LAYERS)
        neurons_text = st.text_input("Neurons per Layer (comma-separated)", value=f"{NUM_HIDDEN_NEURONS},{NUM_HIDDEN_NEURONS}")
        lr = st.number_input(
            "Learning Rate (eta)",
            min_value=0.0001,
            max_value=1.0,
            value=LEARNING_RATE,
            step=0.0005,
            format="%.4f",
        )
        epochs = st.number_input("Number of Epochs (m)", min_value=10, max_value=20000, value=EPOCHS, step=10)
        activation = st.selectbox(
            "Activation Function",
            options=["Sigmoid", "Hyperbolic Tangent"],
            index=1 if TANH_HIDDEN else 0,
        )
        use_bias = st.checkbox("Add Bias", value=BIAS)

    return {
        "hidden_layers": hidden_layers,
        "neurons_text": neurons_text,
        "lr": lr,
        "epochs": epochs,
        "activation": activation,
        "use_bias": use_bias,
    }


def parse_neurons(neurons_raw, n_layers):
    try:
        values = [int(part.strip()) for part in neurons_raw.split(",") if part.strip()]
        if len(values) == 0:
            return None, "Please enter at least one neuron count."
        if any(v <= 0 for v in values):
            return None, "Neuron counts must be positive integers."
        if len(values) < n_layers:
            values = values + [values[-1]] * (n_layers - len(values))
        elif len(values) > n_layers:
            values = values[:n_layers]
        return values, None
    except ValueError:
        return None, "Neurons must be integers separated by commas, e.g. 32,64."


def train_pipeline(config):
    np.random.seed(37)

    df = pd.read_csv("penguins.csv")
    X, y = preprocessing(df)
    X_train, y_train, X_test, y_test = split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLP(
        bias=config["use_bias"],
        features=FEATURES,
        classes=CLASSES,
        tanh_hidden=config["tanh_hidden"],
        num_of_hidden_layers=config["hidden_layers"],
        num_hidden_neurons=config["neurons_per_layer"][0],
    )
    model.train()

    optimizer = optim.Gradient(model.parameters(), lr=config["learning_rate"])
    losses = []

    progress_bar = st.progress(0, text="Training in progress...")

    for epoch in range(config["epochs"]):
        optimizer.zero_grad()

        x = Tensor(X_train)
        y_onehot = np.eye(CLASSES)[y_train]
        target = Tensor(y_onehot)

        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        losses.append(float(loss.data))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, config["epochs"] // 100) == 0 or epoch == config["epochs"] - 1:
            progress = int(((epoch + 1) / config["epochs"]) * 100)
            progress_bar.progress(progress, text=f"Training... {progress}%")
            time.sleep(0.005)

    train_logits = model(Tensor(X_train)).data
    test_logits = model(Tensor(X_test)).data

    train_pred = np.argmax(train_logits, axis=1)
    test_pred = np.argmax(test_logits, axis=1)

    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    cm = confusion_matrix(y_test, test_pred)

    return {
        "model": model,
        "scaler": scaler,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cm": cm,
        "losses": losses,
        "y_test": y_test,
        "test_pred": test_pred,
        "config": config,
    }


def render_results(run_result):
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📊 Results")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Train Accuracy", f"{run_result['train_acc'] * 100:.2f}%")
    with c2:
        st.metric("Test Accuracy", f"{run_result['test_acc'] * 100:.2f}%")

    st.markdown("#### 🔥 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        run_result["cm"],
        annot=True,
        fmt="d",
        cmap="mako",
        cbar=True,
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"],
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix (Test Set)")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.markdown("#### 📉 Loss Curve")
    fig_loss, ax_loss = plt.subplots(figsize=(8, 3.6))
    ax_loss.plot(run_result["losses"], color="#ffd93d", linewidth=2.2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(alpha=0.25)
    st.pyplot(fig_loss)
    plt.close(fig_loss)

    st.markdown('</div>', unsafe_allow_html=True)


def render_prediction(run_result):
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🕊️ Classification (Single Sample)")
    st.caption("Enter 5 values in this order: BillLength, BillDepth, FlipperLength, BodyMass, OriginLocation")

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        f1 = st.number_input("Feature 1", value=40.0, key="f1")
    with s2:
        f2 = st.number_input("Feature 2", value=17.0, key="f2")
    with s3:
        f3 = st.number_input("Feature 3", value=190.0, key="f3")
    with s4:
        f4 = st.number_input("Feature 4", value=3700.0, key="f4")
    with s5:
        f5 = st.number_input("Feature 5", value=1.0, key="f5")

    if st.button("Predict Class ID", use_container_width=True):
        sample = np.array([[f1, f2, f3, f4, f5]], dtype=float)
        sample_scaled = run_result["scaler"].transform(sample)
        pred_logits = run_result["model"](Tensor(sample_scaled)).data
        pred_class = int(np.argmax(pred_logits, axis=1)[0])
        st.success(f"Predicted Class ID: {pred_class}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_history_table():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📚 Experiment Summary Table")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No runs yet. Train at least one model to populate this summary table.")
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    np.random.seed(37)

    setup_page()
    render_header()
    initialize_state()

    ui_config = sidebar_config()

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🚀 Training Section")
    start_training = st.button("Start Training", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    run_result = None
    if start_training:
        neurons_per_layer, parse_error = parse_neurons(ui_config["neurons_text"], ui_config["hidden_layers"])
        if parse_error:
            st.error(parse_error)
        else:
            if len(set(neurons_per_layer)) > 1:
                st.info(
                    "Current backend MLP uses one hidden width value. "
                    "This run will apply the first value to all hidden layers."
                )

            config = {
                "hidden_layers": ui_config["hidden_layers"],
                "neurons_per_layer": neurons_per_layer,
                "learning_rate": float(ui_config["lr"]),
                "epochs": int(ui_config["epochs"]),
                "tanh_hidden": ui_config["activation"] == "Hyperbolic Tangent",
                "use_bias": ui_config["use_bias"],
                "activation_name": ui_config["activation"],
            }

            with st.spinner("🔮 Initializing model and running backpropagation..."):
                run_result = train_pipeline(config)

            st.session_state.last_run = run_result
            st.session_state.history.append(
                {
                    "Activation": ui_config["activation"],
                    "Hidden Layers": ui_config["hidden_layers"],
                    "Neurons": ",".join(map(str, neurons_per_layer)),
                    "Learning Rate": float(ui_config["lr"]),
                    "Epochs": int(ui_config["epochs"]),
                    "Bias": ui_config["use_bias"],
                    "Train Accuracy": round(run_result["train_acc"], 4),
                    "Test Accuracy": round(run_result["test_acc"], 4),
                }
            )

    if "last_run" in st.session_state:
        run_result = st.session_state.last_run

    if run_result is not None:
        render_results(run_result)
        render_prediction(run_result)

    render_history_table()


if __name__ == "__main__":
    main()
