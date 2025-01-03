import streamlit as st
from pathlib import Path
import json
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import librosa
import librosa.display
import asyncio
import aiohttp
import numpy as np
import os

# Конфигурация логгирования
logger = logging.getLogger("StreamlitService")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("logs/streamlit_service.log", maxBytes=5000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# API endpoints
UPLOAD_DATA_URL = "http://backend:8000/api/v1/models/upload_data"
TRAIN_MODEL_URL = "http://backend:8000/api/v1/models/train_model"
MODEL_INFO_URL = "http://backend:8000/api/v1/models/get_model_list"
PREDICT_MODEL_URL = "http://backend:8000/api/v1/models/predict"  
SELECT_MODEL_URL = "http://backend:8000/api/v1/models/set_model"

# Асинхронная функция для обработки аудиофайла
async def run_prediction_and_process(uploaded_audio):
    try:
        logger.info("Загрузка данных для предсказания")
        audio_data, sr = librosa.load(uploaded_audio, sr=None)

        # Waveform
        st.subheader("Waveform")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
        logger.info("Waveform displayed successfully.")

        # Spectrogram
        st.subheader("Spectrogram")
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        st.pyplot(plt)
        logger.info("Spectrogram displayed successfully.")
        print(type(uploaded_audio))
        # Server request
        async with aiohttp.ClientSession() as session:
            logger.info("Запрос предсказания для данного аудио")
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                uploaded_audio.getvalue(),
                filename="uploaded_audio.wav",
                content_type="audio/wav"
            )
            async with session.post(PREDICT_MODEL_URL, data=form_data) as response:
                if response.status == 200:
                    prediction_result = await response.json()
                    st.success("Предсказание получено")
                    st.json(prediction_result)
                    logger.info("Предсказание получено")
                else:
                    error_message = f"Предсказание ушло в {response.status}"
                    st.error(error_message)
                    logger.error(error_message)
    except Exception as e:
        logger.error(f"Ошибка во время предикта {str(e)}")
        st.error(f"Произошла ошибка: {str(e)}")

async def process_dataset(payload=None, file_path=None):
    """
    Если передаём file_path – значит отправляем файл (один).
    payload может быть словарём или строкой (JSON).
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Формируем multipart/form-data
            form_data = aiohttp.FormData()

            # Если есть файл, добавляем
            if file_path is not None:
                logger.info("Отправка датасета с файлом")
                form_data.add_field(
                    name="file",
                    value=files["file"],
                    filename="uploaded_file.npz",
                    content_type="application/octet-stream",
                )

            # Если есть payload, сериализуем в JSON (если это dict/список)
            if payload is not None:
                # Если payload у нас не строка, делаем json.dumps
                if not isinstance(payload, str):
                    payload = json.dumps(payload, ensure_ascii=False)

                # Добавляем в form_data
                form_data.add_field(
                    name='payload',
                    value=payload
                )

            # Делаем POST-запрос
            async with session.post(UPLOAD_DATA_URL, data=form_data) as response:
                # Проверяем статус
                if response.status == 200:
                    logger.info("Отправка завершена успешно")
                    return True, None
                else:
                    error_message = await response.text()
                    logger.error(f"Ошибка при отправке: {error_message}")
                    return False, error_message

    except Exception as e:
        logger.error(f"Ошибка при отправке: {str(e)}")
        return False, str(e)

async def train_model(payload):
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("Отправка модели для обучения")
            async with session.post(TRAIN_MODEL_URL, json=payload) as response:
                if response.status == 200:
                    logger.info("Успешно началось")
                    return True, None
                else:
                    error_message = await response.text()
                    logger.error(f"МОдель не обучилась {error_message}")
                    return False, error_message
    except Exception as e:
        logger.error(f"Ошибка при обучение{str(e)}")
        return False, str(e)

async def fetch_model_info():
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("Информация о модели сюда")
            async with session.get(MODEL_INFO_URL) as response:
                if response.status == 200:
                    logger.info("Получили инфу о модели")
                    return await response.json(), None
                else:
                    error_message = "Ошибка при получении списка моделей"
                    logger.error(error_message)
                    return None, error_message
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей {str(e)}")
        return None, str(e)

async def fetch_model_metrics(selected_model):
    try:
        payload = {"model_name": selected_model}
        async with aiohttp.ClientSession() as session:
            logger.info(f"Получение метрик для модели: {selected_model}")
            async with session.post(f"{MODEL_INFO_URL}/metrics", json=payload) as response:
                if response.status == 200:
                    logger.info(f"ПОлучение метрик для модели: {selected_model}")
                    return await response.json(), None
                else:
                    error_message = await response.text()
                    logger.error(f"не получилось получить метрики для модели {error_message}")
                    return None, error_message
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        return None, str(e)

async def set_model(selected_model):
    try:
        payload = {"model_name": selected_model}
        async with aiohttp.ClientSession() as session:
            logger.info(f"Выбор модели: {selected_model}")
            async with session.post(f"{SELECT_MODEL_URL}", json=payload) as response:
                if response.status == 200:
                    logger.info(f"Выбрана: {selected_model} модель")
                    return await response.json(), None
                else:
                    error_message = await response.text()
                    logger.error(f"не получилось выбрать модель {error_message}")
                    return None, error_message
    except Exception as e:
        logger.error(f"Ошибка при выборе модели: {str(e)}")
        return None, str(e)


# Инициализация состояния
if "page" not in st.session_state:
    st.session_state.page = "Формирование датасета"

# Навигация в боковой панели
st.sidebar.title("Навигация")
option = st.sidebar.radio(
    "Выберите страницу:",
    ["Формирование датасета", "Обучение модели", "Информация о модели", "Использование модели"],
    index=["Формирование датасета", "Обучение модели", "Информация о модели", "Использование модели"].index(st.session_state.page),
)
st.session_state.page = option

# Отображение выбранной страницы
if st.session_state.page == "Формирование датасета":
    st.header("Формирование датасета")
    uploaded_npz = st.file_uploader("Загрузите .npz файл модели", type=["npz"])
    st.markdown("**ИЛИ**")
    mp3_vocals_root = st.text_input("MP3 Vocals Root Directory", placeholder="/path/to/MP3_VOCALS_DIR")
    lmd_aligned_vocals_root = st.text_input("LMD Aligned Vocals Directory", placeholder="/path/to/MIDI_VOCALS_DIR")
    match_scores_json = st.text_input("Match Scores JSON Path", placeholder="/path/to/MATCH_SCORES_PATH")
    output_npz = st.text_input("Output NPZ Path", placeholder="/path/to/dataset.npz")

    if st.button("Запустить формирование датасета"):
        if uploaded_npz is None and (not mp3_vocals_root or not lmd_aligned_vocals_root or not match_scores_json or not output_npz):
            st.error("Необходимо заполнить первую или вторую форму")
        else:
            payload = {
                "mp3_vocals_root": mp3_vocals_root,
                "lmd_aligned_vocals_root": lmd_aligned_vocals_root,
                "match_scores_json": match_scores_json,
                "output_npz": output_npz,
            }
            files = {"file": uploaded_npz.getvalue()} if uploaded_npz else None
            success, error = asyncio.run(process_dataset(payload, files))
            if success:
                st.success("Формирование датасета выполнено успешно")
            else:
                st.error(f"Ошибка: {error}")

elif st.session_state.page == "Обучение модели":
    st.header("Обучение модели")
    st.write("Заполните параметры для обучения модели:")

    id_param = st.number_input("ID модели", value=0, step=1, format="%d")
    n_jobs = st.number_input("Количество потоков (n_jobs)", value=0, step=1, format="%d")
    penalty = st.selectbox("Тип штрафа (penalty)", options=["l1", "l2", "elasticnet", "none"], index=1)
    loss = st.selectbox("Тип функции потерь (loss)", options=["hinge", "squared_hinge", "log"], index=0)

    hyperparameters_json = st.text_area(
        "Дополнительные гиперпараметры (JSON формат, опционально)", value="{}", height=100
    )

    if st.button("Обучение"):
        try:
            hyperparameters = json.loads(hyperparameters_json)
            logger.info("Parsed hyperparameters successfully.")
        except json.JSONDecodeError:
            st.error("Неверный формат JSON для дополнительных гиперпараметров.")
            logger.error("Invalid JSON format for additional hyperparameters.")
            hyperparameters = {}

        payload = {
            "id": id_param,
            "hyperparameters": hyperparameters,
            "n_jobs": n_jobs,
            "penalty": penalty,
            "loss": loss,
        }

        success, error = asyncio.run(train_model(payload))
        if success:
            st.success(f"Модель успешно обучена!, ответ сервера: {success}")
        else:
            st.error(f"Ошибка обучения модели: {error}")

elif st.session_state.page == "Информация о модели":
    st.header("Информация о модели")
    models, error = asyncio.run(fetch_model_info())
    if error:
        st.error(f"Ошибка при получении списка моделей: {error}")
    elif models:
        selected_model = st.selectbox("Выберите модель", options=models.get("message", []))
        if selected_model:
            metrics, error = asyncio.run(fetch_model_metrics(selected_model))
            if error:
                st.error(f"Ошибка при получении метрик модели: {error}")
            elif metrics:
                if "loss" in metrics:
                    st.subheader("График лосса")
                    st.line_chart(metrics["loss"])
                else:
                    st.warning("Нет данных о лоссе для данной модели")
                if "accuracy" in metrics:
                    st.subheader("График аккураси")
                    st.line_chart(metrics["accuracy"])
                else:
                    st.warning("Нет данных об аккураси для данной модели")
    else:
        st.warning("Список моделей пуст. Проверьте доступность API или загрузите модели.")

elif st.session_state.page == "Использование модели":
    st.header("Использование модели")
    uploaded_audio = st.file_uploader("Загрузка аудио", type=["wav"])
    st.header('Выберите модель из списка, которую хотите использовать')
    models, error = asyncio.run(fetch_model_info())
    if error:
        st.error(f"Ошибка при получении списка моделей: {error}")
    elif models:
        selected_model = st.selectbox("Выберите модель", options=models.get("message", []))
    models, error = asyncio.run(set_model(selected_model=selected_model))
    if error:
        st.error(f'ошибка выора модели: {error}')
    else:
        st.success('Модель успешно выбрана')
    btn_predict = st.button("Начать обработку")
    if uploaded_audio and btn_predict:
        asyncio.run(run_prediction_and_process(uploaded_audio))
    elif not uploaded_audio:
        st.error('Не загрузили аудио')
