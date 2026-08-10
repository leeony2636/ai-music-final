# ============================================================
# AI 음악 장르 예측기
# 6강 기본 / 7강 고급 / 8강 ResNet18
# ============================================================

import os
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 음악 장르 예측기",
    page_icon="🎵",
    layout="wide"
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# 1. 사이드바
# ============================================================

st.sidebar.title("🎵 AI 음악")

page = st.sidebar.radio(
    "강의 선택",
    [
        "6강 · 기본 예측",
        "7강 · 고급 분석",
        "8강 · ResNet18"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "GTZAN 10개 음악 장르 분류"
)

st.sidebar.write(
    "6강 → 머신러닝 기본"
)

st.sidebar.write(
    "7강 → 신뢰도·멀티파일"
)

st.sidebar.write(
    "8강 → ResNet18 딥러닝"
)


# ============================================================
# 2. 6·7강 RandomForest 모델
# ============================================================

FEATURE_COLS = (
    [
        "chroma_stft_mean",
        "chroma_stft_var",
        "rms_mean",
        "rms_var",
        "spectral_centroid_mean",
        "spectral_centroid_var",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_var",
        "rolloff_mean",
        "rolloff_var",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_var",
        "harmony_mean",
        "harmony_var",
        "perceptr_mean",
        "perceptr_var",
        "tempo"
    ]
    +
    [
        f"mfcc{i}_{s}"
        for i in range(1, 21)
        for s in ("mean", "var")
    ]
)


@st.cache_resource
def load_rf_models():

    rf_path = os.path.join(
        BASE_DIR,
        "models",
        "model_rf.joblib"
    )

    le_path = os.path.join(
        BASE_DIR,
        "models",
        "label_encoder.joblib"
    )

    sc_path = os.path.join(
        BASE_DIR,
        "models",
        "scaler.joblib"
    )

    rf = joblib.load(rf_path)
    le = joblib.load(le_path)
    sc = joblib.load(sc_path)

    return rf, le, sc


def extract_rf_features(wav_path):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True,
        duration=3.0
    )

    feats = {}

    # Chroma
    ch = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    feats["chroma_stft_mean"] = float(
        np.mean(ch)
    )

    feats["chroma_stft_var"] = float(
        np.var(ch)
    )

    # RMS
    rms = librosa.feature.rms(
        y=y
    )

    feats["rms_mean"] = float(
        np.mean(rms)
    )

    feats["rms_var"] = float(
        np.var(rms)
    )

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    feats["spectral_centroid_mean"] = float(
        np.mean(sc)
    )

    feats["spectral_centroid_var"] = float(
        np.var(sc)
    )

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    feats["spectral_bandwidth_mean"] = float(
        np.mean(bw)
    )

    feats["spectral_bandwidth_var"] = float(
        np.var(bw)
    )

    # Rolloff
    ro = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    feats["rolloff_mean"] = float(
        np.mean(ro)
    )

    feats["rolloff_var"] = float(
        np.var(ro)
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    feats["zero_crossing_rate_mean"] = float(
        np.mean(zcr)
    )

    feats["zero_crossing_rate_var"] = float(
        np.var(zcr)
    )

    # Harmonic / Percussive
    harm, perc = librosa.effects.hpss(
        y
    )

    feats["harmony_mean"] = float(
        np.mean(harm)
    )

    feats["harmony_var"] = float(
        np.var(harm)
    )

    feats["perceptr_mean"] = float(
        np.mean(perc)
    )

    feats["perceptr_var"] = float(
        np.var(perc)
    )

    # Tempo
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    if np.ndim(tempo) == 0:
        feats["tempo"] = float(tempo)
    else:
        feats["tempo"] = float(tempo[0])

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    for i in range(20):

        feats[f"mfcc{i+1}_mean"] = float(
            np.mean(mfcc[i])
        )

        feats[f"mfcc{i+1}_var"] = float(
            np.var(mfcc[i])
        )

    vector = np.array(
        [
            feats[col]
            for col in FEATURE_COLS
        ],
        dtype=np.float32
    )

    return vector.reshape(
        1,
        -1
    )


def predict_rf(
    wav_path,
    rf,
    le,
    scaler
):

    features = extract_rf_features(
        wav_path
    )

    features_scaled = scaler.transform(
        features
    )

    probabilities = rf.predict_proba(
        features_scaled
    )[0]

    prob_dict = {
        le.classes_[i]:
        float(probabilities[i])

        for i in range(
            len(probabilities)
        )
    }

    prob_dict = dict(
        sorted(
            prob_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    top_genre = next(
        iter(prob_dict)
    )

    return (
        top_genre,
        prob_dict
    )


# ============================================================
# 3. 8강 ResNet18 최종 모델
# ============================================================

@st.cache_resource
def load_resnet_model():

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "resnet18_gtzan_final.pth"
    )

    model = resnet18(
        weights=None
    )

    # 학습할 때 사용한 최종 FC 구조
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            512,
            10
        )
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


resnet_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ]
)


# ============================================================
# 4. WAV → 3초 Mel Spectrogram
# ============================================================

def wav_segment_to_image(
    y,
    sr=22050
):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # 0 ~ 255 정규화
    mel_min = mel_db.min()
    mel_max = mel_db.max()

    mel_norm = (
        (mel_db - mel_min)
        /
        (mel_max - mel_min + 1e-8)
    )

    mel_uint8 = (
        mel_norm * 255
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        mel_uint8
    ).convert(
        "RGB"
    )

    return image


# ============================================================
# 5. ResNet 예측
# 전체 WAV를 3초씩 나눈 뒤 평균
# ============================================================

def predict_resnet(
    wav_path,
    model
):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True
    )

    segment_length = (
        sr * 3
    )

    tensors = []

    # 3초 단위 분할
    for start in range(
        0,
        len(y),
        segment_length
    ):

        segment = y[
            start:
            start + segment_length
        ]

        # 너무 짧은 마지막 조각 제외
        if len(segment) < sr:
            continue

        # 3초보다 짧으면 padding
        if len(segment) < segment_length:

            segment = librosa.util.fix_length(
                segment,
                size=segment_length
            )

        image = wav_segment_to_image(
            segment,
            sr
        )

        tensor = resnet_transform(
            image
        )

        tensors.append(
            tensor
        )

    # 혹시 짧은 WAV라면
    if not tensors:

        y = librosa.util.fix_length(
            y,
            size=segment_length
        )

        image = wav_segment_to_image(
            y,
            sr
        )

        tensors.append(
            resnet_transform(
                image
            )
        )

    batch = torch.stack(
        tensors
    ).to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            batch
        )

        # 각 3초 구간의 결과 평균
        mean_logits = logits.mean(
            dim=0
        )

        probabilities = torch.softmax(
            mean_logits,
            dim=0
        ).cpu().numpy()

    prob_dict = {
        GENRES[i]:
        float(probabilities[i])

        for i in range(
            len(GENRES)
        )
    }

    prob_dict = dict(
        sorted(
            prob_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    top_genre = next(
        iter(prob_dict)
    )

    return (
        top_genre,
        prob_dict,
        len(tensors)
    )


# ============================================================
# 6. 공통 그래프
# ============================================================

def plot_probability(
    prob_dict,
    title=""
):

    genres = list(
        prob_dict.keys()
    )

    probs = list(
        prob_dict.values()
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.barh(
        genres[::-1],
        probs[::-1]
    )

    for i, prob in enumerate(
        probs[::-1]
    ):

        ax.text(
            prob + 0.01,
            i,
            f"{prob:.1%}",
            va="center",
            fontsize=9
        )

    ax.set_xlim(
        0,
        1.1
    )

    ax.set_xlabel(
        "확률"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    return fig


def plot_melspectrogram(
    wav_path,
    title=""
):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True,
        duration=10.0
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    ax.set_title(
        title or "Mel Spectrogram"
    )

    fig.tight_layout()

    return fig


# ============================================================
# 7. 임시 WAV 저장
# ============================================================

def save_uploaded_wav(
    uploaded_file
):

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    tmp.write(
        uploaded_file.getvalue()
    )

    tmp.close()

    return tmp.name


# ============================================================
# 8. 6강 기본
# ============================================================

if page == "6강 · 기본 예측":

    st.title(
        "🎵 6강 · 음악 장르 예측기"
    )

    st.caption(
        "RandomForest 기반 기본 장르 예측"
    )

    try:

        rf_model, le_model, scaler_model = (
            load_rf_models()
        )

        st.success(
            "6강 모델 로드 완료"
        )

    except Exception as e:

        st.error(
            f"6강 모델을 불러올 수 없습니다: {e}"
        )

        st.stop()

    uploaded_file = st.file_uploader(
        "WAV 파일 업로드",
        type=["wav"],
        key="lesson6"
    )

    if uploaded_file:

        wav_path = save_uploaded_wav(
            uploaded_file
        )

        st.audio(
            uploaded_file
        )

        with st.spinner(
            "장르 분석 중..."
        ):

            top_genre, probabilities = predict_rf(
                wav_path,
                rf_model,
                le_model,
                scaler_model
            )

        st.subheader(
            "예측 결과"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "예측 장르",
                top_genre.upper()
            )

        with col2:

            st.metric(
                "신뢰도",
                f"{probabilities[top_genre]:.1%}"
            )

        fig = plot_probability(
            probabilities,
            "장르별 예측 확률"
        )

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        st.subheader(
            "멜스펙트로그램"
        )

        fig = plot_melspectrogram(
            wav_path,
            uploaded_file.name
        )

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        os.unlink(
            wav_path
        )


# ============================================================
# 9. 7강 고급
# ============================================================

elif page == "7강 · 고급 분석":

    st.title(
        "🎵 7강 · 고급 음악 분석"
    )

    st.caption(
        "멀티파일 비교 · 신뢰도 · 예측 이력"
    )

    try:

        rf_model, le_model, scaler_model = (
            load_rf_models()
        )

        st.success(
            "7강 모델 로드 완료"
        )

    except Exception as e:

        st.error(
            f"7강 모델을 불러올 수 없습니다: {e}"
        )

        st.stop()

    if "history7" not in st.session_state:

        st.session_state[
            "history7"
        ] = []

    tab_predict, tab_history = st.tabs(
        [
            "🎵 예측",
            "📋 이력"
        ]
    )

    with tab_predict:

        uploaded_files = st.file_uploader(
            "WAV 파일 업로드 (여러 개 가능)",
            type=["wav"],
            accept_multiple_files=True,
            key="lesson7"
        )

        if uploaded_files:

            result_rows = []

            for uploaded_file in uploaded_files:

                wav_path = save_uploaded_wav(
                    uploaded_file
                )

                top_genre, probabilities = predict_rf(
                    wav_path,
                    rf_model,
                    le_model,
                    scaler_model
                )

                second_genre = list(
                    probabilities.keys()
                )[1]

                result_rows.append(
                    {
                        "파일명":
                        uploaded_file.name,

                        "1위 장르":
                        top_genre,

                        "1위 확률":
                        f"{probabilities[top_genre]:.1%}",

                        "2위 장르":
                        second_genre
                    }
                )

                st.session_state[
                    "history7"
                ].append(
                    result_rows[-1]
                )

                if len(uploaded_files) == 1:

                    st.audio(
                        uploaded_file
                    )

                    col1, col2 = st.columns(
                        2
                    )

                    with col1:

                        st.subheader(
                            "신뢰도 분포"
                        )

                        fig = plot_probability(
                            probabilities,
                            uploaded_file.name
                        )

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

                    with col2:

                        st.subheader(
                            "멜스펙트로그램"
                        )

                        fig = plot_melspectrogram(
                            wav_path,
                            uploaded_file.name
                        )

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

                os.unlink(
                    wav_path
                )

            if len(
                uploaded_files
            ) > 1:

                st.subheader(
                    "멀티파일 비교"
                )

                st.dataframe(
                    pd.DataFrame(
                        result_rows
                    ),
                    use_container_width=True
                )

    with tab_history:

        st.subheader(
            "예측 이력"
        )

        if st.session_state[
            "history7"
        ]:

            st.dataframe(
                pd.DataFrame(
                    st.session_state[
                        "history7"
                    ]
                ),
                use_container_width=True
            )

            if st.button(
                "이력 초기화"
            ):

                st.session_state[
                    "history7"
                ] = []

                st.rerun()

        else:

            st.info(
                "아직 예측 이력이 없습니다."
            )


# ============================================================
# 10. 8강 ResNet18
# ============================================================

elif page == "8강 · ResNet18":

    st.title(
        "🧠 8강 · ResNet18 음악 장르 예측"
    )

    st.caption(
        "3초 Mel Spectrogram + ResNet18 최종 학습 모델"
    )

    try:

        resnet_model = load_resnet_model()

        st.success(
            "최종 ResNet18 모델 로드 완료"
        )

    except Exception as e:

        st.error(
            f"ResNet 모델을 불러올 수 없습니다: {e}"
        )

        st.stop()


    # ========================================================
    # AS-IS / TO-BE
    # ========================================================

    st.subheader(
        "📊 AS-IS / TO-BE 모델 비교"
    )

    lesson8_col_as_is, lesson8_col_to_be = st.columns(2)

    with lesson8_col_as_is:

        st.markdown(
            "### AS-IS"
        )

        st.metric(
            "기존 ResNet18 정확도",
            "76.47%"
        )

        st.caption(
            "Epoch 10 · Batch Size 64"
        )


    with lesson8_col_to_be:

        st.markdown(
            "### TO-BE"
        )

        st.metric(
            "최종 ResNet18 정확도",
            "82.32%",
            delta="+5.85%p"
        )

        st.caption(
            "데이터 증강 및 하이퍼파라미터 조정 적용"
        )


    # ========================================================
    # Epoch별 Accuracy / Loss 데이터
    # ========================================================

    st.subheader(
        "📈 Epoch별 학습 변화"
    )

    lesson8_training_history_df = pd.DataFrame({

        "Epoch": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        ],

        "Validation Accuracy": [
            45.99,
            63.10,
            66.31,
            74.33,
            74.33,
            75.94,
            74.33,
            75.94,
            77.54,
            76.47
        ],

        "Train Loss": [
            2.0178,
            1.3673,
            1.1178,
            0.9095,
            0.8398,
            0.7478,
            0.7008,
            0.6821,
            0.6658,
            0.6410
        ]
    })


    # ========================================================
    # Accuracy + Loss 그래프
    # ========================================================

    lesson8_fig_training, lesson8_ax_accuracy = plt.subplots(
        figsize=(7, 3.5)
    )

    lesson8_ax_accuracy.plot(
        lesson8_training_history_df["Epoch"],
        lesson8_training_history_df["Validation Accuracy"],
        color="tab:blue",
        marker="o",
        linewidth=2,
        markersize=4,
        label="Validation Accuracy"
    )

    lesson8_ax_accuracy.set_xlabel(
        "Epoch"
    )

    lesson8_ax_accuracy.set_ylabel(
        "Validation Accuracy (%)",
        color="tab:blue"
    )

    lesson8_ax_accuracy.tick_params(
        axis="y",
        labelcolor="tab:blue"
    )

    lesson8_ax_accuracy.set_xticks(
        lesson8_training_history_df["Epoch"]
    )

    lesson8_ax_accuracy.grid(
        alpha=0.2
    )


    # 오른쪽 Y축 : Train Loss
    lesson8_ax_loss = lesson8_ax_accuracy.twinx()

    lesson8_ax_loss.plot(
        lesson8_training_history_df["Epoch"],
        lesson8_training_history_df["Train Loss"],
        color="tab:orange",
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=4,
        label="Train Loss"
    )

    lesson8_ax_loss.set_ylabel(
        "Train Loss",
        color="tab:orange"
    )

    lesson8_ax_loss.tick_params(
        axis="y",
        labelcolor="tab:orange"
    )

    lesson8_fig_training.suptitle(
        "Epoch별 Validation Accuracy / Train Loss",
        fontsize=11
    )

    lesson8_fig_training.tight_layout()

    st.pyplot(
        lesson8_fig_training,
        use_container_width=False
    )

    plt.close(
        lesson8_fig_training
    )


    # ========================================================
    # Learning Rate 데이터
    # ========================================================

    lesson8_learning_rate_df = pd.DataFrame({

        "Epoch": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        ],

        "Learning Rate": [
            0.00010000,
            0.00009758,
            0.00009055,
            0.00007960,
            0.00006580,
            0.00005050,
            0.00003520,
            0.00002140,
            0.00001045,
            0.00000342
        ]
    })


    # ========================================================
    # Learning Rate 그래프
    # ========================================================

    st.subheader(
        "⚙️ Epoch별 Learning Rate 변화"
    )

    lesson8_fig_lr, lesson8_ax_lr = plt.subplots(
        figsize=(7, 2.8)
    )

    lesson8_ax_lr.plot(
        lesson8_learning_rate_df["Epoch"],
        lesson8_learning_rate_df["Learning Rate"],
        color="tab:green",
        marker="o",
        linewidth=2,
        markersize=4
    )

    lesson8_ax_lr.set_xlabel(
        "Epoch"
    )

    lesson8_ax_lr.set_ylabel(
        "Learning Rate",
        color="tab:green"
    )

    lesson8_ax_lr.tick_params(
        axis="y",
        labelcolor="tab:green"
    )

    lesson8_ax_lr.set_xticks(
        lesson8_learning_rate_df["Epoch"]
    )

    lesson8_ax_lr.grid(
        alpha=0.2
    )

    lesson8_ax_lr.set_title(
        "Epoch별 Learning Rate",
        fontsize=11
    )

    lesson8_fig_lr.tight_layout()

    st.pyplot(
        lesson8_fig_lr,
        use_container_width=False
    )

    plt.close(
        lesson8_fig_lr
    )


    st.caption(
        "기존 ResNet18 76.47% → "
        "최종 ResNet18 82.32% "
        "(+5.85%p)"
    )


    # ========================================================
    # WAV 파일 업로드 및 최종 모델 예측
    # ========================================================

    uploaded_file = st.file_uploader(
        "WAV 파일 업로드",
        type=["wav"],
        key="lesson8"
    )

    if uploaded_file:

        wav_path = save_uploaded_wav(
            uploaded_file
        )

        st.audio(
            uploaded_file
        )

        with st.spinner(
            "3초 구간으로 나누어 ResNet18 분석 중..."
        ):

            (
                top_genre,
                probabilities,
                segment_count
            ) = predict_resnet(
                wav_path,
                resnet_model
            )

        st.success(
            "분석 완료"
        )


        # ====================================================
        # 예측 결과
        # ====================================================

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "예측 장르",
                top_genre.upper()
            )

        with col2:

            st.metric(
                "예측 확률",
                f"{probabilities[top_genre]:.1%}"
            )

        with col3:

            st.metric(
                "분석 구간",
                f"{segment_count}개"
            )


        # ====================================================
        # 확률 / 멜스펙트로그램
        # ====================================================

        col_left, col_right = st.columns(
            2
        )

        with col_left:

            st.subheader(
                "장르별 확률"
            )

            fig = plot_probability(
                probabilities,
                "ResNet18 예측 확률"
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


        with col_right:

            st.subheader(
                "멜스펙트로그램"
            )

            fig = plot_melspectrogram(
                wav_path,
                uploaded_file.name
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


        st.caption(
            "전체 음악을 3초 단위로 분석한 뒤 "
            "각 구간의 ResNet18 결과를 평균하여 "
            "최종 장르를 결정합니다."
        )

        os.unlink(
            wav_path
        )
