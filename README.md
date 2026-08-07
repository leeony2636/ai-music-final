# 🎵 AI Music Genre Classifier

음악 파일을 업로드하면 AI 모델이 음악의 장르를 예측하는 프로젝트입니다.

Random Forest와 ResNet18 모델을 활용하여 음악 데이터를 분석하고,
예측 결과와 장르별 확률을 확인할 수 있도록 구현했습니다.

---

## 📌 프로젝트 소개

음악의 특징을 분석하여 장르를 자동으로 분류하는 AI 프로젝트입니다.

기존 머신러닝 모델과 딥러닝 모델을 활용하면서
음악 데이터 전처리부터 모델 학습, 예측까지의 과정을 실습했습니다.

---

## 🤖 사용 모델

| 모델 | 설명 |
|---|---|
| Random Forest | 음악 특징 데이터를 이용한 기본 장르 분류 모델 |
| ResNet18 | Mel-Spectrogram 이미지를 이용한 딥러닝 장르 분류 모델 |

---

## 🎼 주요 기능

- 음악 파일 업로드
- 음악 특징 추출
- AI 음악 장르 예측
- 장르별 예측 확률 확인
- Mel-Spectrogram 시각화
- Random Forest / ResNet18 모델 활용

---
## ⚙️ 실행 환경

- Python 3.x
- VS Code
- Streamlit
---
## 📦 필요한 패키지

프로젝트 실행에 필요한 Python 패키지는 `requirements.txt`에 정리되어 있습니다.

아래 명령어로 한 번에 설치할 수 있습니다.

```bash
pip install -r requirements.txt
---

## 📂 프로젝트 구조

```text
AI-Music/
│
├── app.py
├── model_rf.joblib
├── resnet18_gtzan_final.pth
├── label_encoder.joblib
├── scaler.joblib
├── requirements.txt
└── README.md
