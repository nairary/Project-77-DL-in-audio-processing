import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import json
import math
import numpy as np
import librosa
import pretty_midi
import joblib
from sklearn.utils.class_weight import compute_class_weight
from fastapi import UploadFile, File
from typing import List, Dict, Any
from sklearn.linear_model import SGDClassifier

from settings.config import (MODELS_DIR, FEATURES_DIR, PREDICTIONS_DIR)
from serializers.serializers import (FitRequest)

# константы
SR = 22050
HOP_LENGTH = 512
CONTEXT_SIZE = 10
N_MELS = 80
SILENCE_PITCH = -1

CURRENT_DIRECTORY = os.getcwd()

################################ EXTRACT FEATURES ################################

def extract_features(audio: List[float], sr: int) -> List[List[float]]:
    """
    Извлекает фичи:
     - Mel-спектр (в dB)
     - Сумма амплитуд (frame_energy)
     - Контекст ±CONTEXT_SIZE
    """
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=1.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_T = S_db.T
    num_frames = S_db_T.shape[0]

    # Линейная версия для энергии
    S_lin = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=1.0
    )
    frame_energy = np.sum(S_lin, axis=0)  # (num_frames,)

    feats_no_context = np.hstack([
        S_db_T,
        frame_energy[:, None]
    ])

    # Добавляем контекст
    feat_with_context = []
    for i in range(num_frames):
        frames_stack = []
        for offset in range(-CONTEXT_SIZE, CONTEXT_SIZE+1):
            idx = i + offset
            if idx < 0:
                idx = 0
            if idx >= num_frames:
                idx = num_frames - 1
            frames_stack.append(feats_no_context[idx])
        stacked = np.concatenate(frames_stack, axis=0)
        feat_with_context.append(stacked)

    return np.array(feat_with_context)


def midi_to_class(midi_pitch: int) -> int:
    if midi_pitch == SILENCE_PITCH:
        return 0
    
    return midi_pitch + 1

def generate_labels_from_vocal_midi(vocals_wav: str, midi_path: str, collision_resolver: str = 'max') -> List[int]:
    """
    Генерирует пофреймовые метки.
    При коллизиях оставляем более низкую ноту.
    """
    audio, _ = librosa.load(vocals_wav, sr=SR)
    num_samples = len(audio)
    num_frames = math.ceil(num_samples / HOP_LENGTH)

    labels = np.full(shape=(num_frames,), fill_value=-1, dtype=int)

    pm = pretty_midi.PrettyMIDI(midi_path)
    if len(pm.instruments) == 0:
        return labels

    inst = pm.instruments[0]
    for note in inst.notes:
        pitch = note.pitch
        start_sec = note.start
        end_sec   = note.end

        start_frame = int((start_sec * SR) // HOP_LENGTH)
        end_frame   = int((end_sec   * SR) // HOP_LENGTH)

        if start_frame < 0:
            start_frame = 0
        if end_frame >= num_frames:
            end_frame = num_frames - 1

        for f_idx in range(start_frame, end_frame+1):
            current_pitch = labels[f_idx]
            if current_pitch == -1:
                labels[f_idx] = pitch
            else:
                if collision_resolver == 'max':
                    labels[f_idx] = max(current_pitch, pitch)
                elif collision_resolver == 'min':
                    labels[f_idx] = min(current_pitch, pitch)

    return labels

def extract_and_save_data(mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz, collision_resolver='min') -> None:
    """
    1) Для каждого MSD ID ищет vocals.wav
    2) Находит MIDI (из lmd_aligned_vocals по best_md5),
    3) Генерирует labels (generate_labels_from_vocal_midi),
       извлекает фичи (extract_features),
       приводит метки к классам (midi_to_class).
    4) Склеивает все X, Y
    5) Сохраняет X, Y в .npz (или .pkl), чтобы не делать
       этот этап при каждом обучении.
    """

    global MP3_VOCALS_DIR
    global MIDI_VOCALS_DIR
    global MATCH_SCORES_PATH
    global FEATURES_DIR

    MP3_VOCALS_DIR = mp3_vocals_root
    MIDI_VOCALS_DIR = lmd_aligned_vocals_root
    MATCH_SCORES_PATH = match_scores_json
    FEATURES_DIR = output_npz + ".npz"
    

    with open(MATCH_SCORES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Сопоставляем MSD ID → best_md5
    best_md5_map = {}
    for msd_id, md5_scores in data.items():
        best_md5 = None
        best_score = float('-inf')
        for m, sc in md5_scores.items():
            if sc > best_score:
                best_score = sc
                best_md5 = m
        best_md5_map[msd_id] = best_md5

    msd_ids = []
    for d in os.listdir(MP3_VOCALS_DIR):
        if d.startswith("TR") and os.path.isdir(os.path.join(MP3_VOCALS_DIR, d)):
            msd_ids.append(d)
            
    X_all, Y_all = [], []

    for msd_id in msd_ids:
        vocals_dir = os.path.join(MP3_VOCALS_DIR, msd_id)
        vocals_wav = os.path.join(vocals_dir, "vocals.wav")
        if not os.path.isfile(vocals_wav):
            print(f"[SKIP] No vocals.wav for {msd_id}")
            continue

        best_md5 = best_md5_map.get(msd_id, None)
        if not best_md5:
            print(f"[SKIP] No best_md5 for {msd_id}")
            continue

        sub1, sub2, sub3 = msd_id[2], msd_id[3], msd_id[4]
        midi_path = os.path.join(
            MIDI_VOCALS_DIR,
            sub1, sub2, sub3,
            msd_id,
            best_md5 + ".mid"
        )
        if not os.path.isfile(midi_path):
            print(f"[SKIP] No MIDI file for {msd_id}")
            continue

        # Генерируем labels
        labels = generate_labels_from_vocal_midi(vocals_wav, midi_path, collision_resolver)

        # Извлекаем фичи
        audio, _ = librosa.load(vocals_wav, sr=SR)
        feats = extract_features(audio, SR)

        min_len = min(len(labels), feats.shape[0])
        labels = labels[:min_len]
        feats  = feats[:min_len]

        # Перевод pitch -> класс (mode)
        class_labels = [midi_to_class(p) for p in labels]

        X_all.append(feats)
        Y_all.append(class_labels)

        print(f"[OK] {msd_id}, frames={min_len}")

    if len(X_all) == 0:
        print("[WARN] No data found to save.")
        return

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print(f"[INFO] Saving dataset: X={X_all.shape}, Y={Y_all.shape} to {FEATURES_DIR}")
    np.savez(FEATURES_DIR, X=X_all, Y=Y_all, allow_pickle=False)
    print("[INFO] Done saving dataset.")

################################ FIT ################################

def sgd_train_iterative(X: List[List[float]], y: List[float], n_epochs: int = 10, batch_size: int = 1000, hyperparameters: Dict[str, Any] = {}) -> SGDClassifier:
    """
    Итеративно обучает SGDClassifier на X, y, 
    разбивая данные на mini-batches (batch_size).
    
    Параметры:
      X, y        - матрица признаков и метки (numpy arrays)
      n_epochs    - сколько эпох (полных проходов по данным)
      batch_size  - размер мини-батча
      n_jobs      - кол-во потоков (ядер); -1 => использовать все
    
    Возвращает обученный SGDClassifier.
    """

    classes_ = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes_, y=y)
    class_weight_dict = {cls: weight for cls, weight in zip(classes_, class_weights)}
    print(f"[INFO] Class weights: {class_weight_dict}")

    # Создаём модель. Параметры для примера:
    clf = SGDClassifier(
        loss=hyperparameters.get("loss", "log_loss"),
        penalty=hyperparameters.get("penalty", "l2"),
        max_iter=2,         # Мы будем сами управлять циклами (n_epochs)
        class_weight=class_weight_dict,
        warm_start=True,     # Чтобы обучаться итеративно, не сбрасывая веса
        shuffle=False,       # Будем сами решать, хотим ли перемешивать
        n_jobs=hyperparameters.get("n_jobs", -1)       # Использовать несколько ядер
    )
    
    # Инициализируем модель "нулевым" fit (чтобы задать параметры и создать структуру)
    # Берём небольшой кусочек данных. Или можно просто fit на одном mini-batch,
    # иначе partial_fit требует обязательно указывать classes=...
    first_batch_size = min(batch_size, X.shape[0])
    
    X_init = X[:first_batch_size]
    y_init = y[:first_batch_size]
    clf.partial_fit(X_init, y_init, classes=classes_)

    # Для каждой эпохи (n_epochs)
    n_samples = X.shape[0]
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        
        # Можно перемешивать порядок индексов, чтобы батчи шли вразнобой
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Проходимся по mini-batches
        batch_count = 0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Обучаем на этом кусочке
            clf.partial_fit(X_batch, y_batch)
            
            batch_count += 1
            # Можно печатать, на каком батче мы находимся
            # Но если batch_size маленький, это будет много вывода
            # print(f"  Batch {batch_count} done.")
        
        # После окончания эпохи печатаем сообщение
        print(f"Epoch {epoch+1} complete. Processed ~{batch_count} mini-batches.")

    return clf

def train_baseline_model_from_npz(input_npz: str, hyperparameters: Dict[str, Any]) -> SGDClassifier:
    """
    1) Загружает X, Y из .npz
    2) Обучает LogisticRegression (или другую модель).
    3) Возвращает модель.
    """

    npz_files = [f for f in os.listdir(input_npz) if f.endswith('.npz')]
    first_npz_file = os.path.join(input_npz, npz_files[0])
    print(first_npz_file)
    data = np.load(first_npz_file)
    X_all = data["X"]
    Y_all = data["Y"]
    print(f"[INFO] Loaded dataset from {input_npz}, X={X_all.shape}, Y={Y_all.shape}")

    clf = sgd_train_iterative(X_all, Y_all, n_epochs=10, batch_size=2000, hyperparameters=hyperparameters.dict())
    clf.fit(X_all, Y_all)
    print(f"[INFO] Model trained (X={X_all.shape}, Y={Y_all.shape})")
    return clf

################################ PREDICT ################################

def predict_pitch_sequence(model: SGDClassifier, audio: List[float], sr: int) -> List[int]:
    """
    Делает предсказание по фреймам (extract_features(audio, sr))
    Возвращает y_pred (num_frames,), где -1 = тишина, иначе MIDI pitch [0..127].
    """
    feats = extract_features(audio, sr)
    y_pred_class = model.predict(feats)

    y_pred_midi = []
    for c in y_pred_class:
        if c == 0:
            y_pred_midi.append(-1)  # тишина
        else:
            y_pred_midi.append(c - 1)
            
    return np.array(y_pred_midi)

def midi_from_prediction(y_pred: List[int], hop_length: int, sr: int, out_midi_path: str = "pred.mid") -> None:
    """
    Создаёт MIDI-файл из массива y_pred (где -1 = тишина, иначе MIDI pitch),
    предполагая, что каждая рамка длится hop_length / sr секунд.

    Логика формирования нот:
      - Если y_pred[i] == pitch (>=0) — значит, в этом фрейме звучит нота pitch.
      - Если pitch не меняется от кадра к кадру, считаем, что это одна «продолжительная» нота.
      - Если меняется (или становится -1), предыдущая нота заканчивается, начинается тишина.

    Сохраняем результат в out_midi_path.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="PredictedVocal")  # program=0 => Acoustic Grand Piano

    frame_duration = hop_length / sr

    current_pitch = -1
    note_on_time = 0.0
    # Обход всех фреймов
    for i, pitch in enumerate(y_pred):
        time_sec = i * frame_duration

        if pitch != current_pitch:
            # Закрываем предыдущую ноту, если она была
            if current_pitch != -1:
                # Добавляем ноту: current_pitch, start=note_on_time, end=time_sec
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_pitch,
                    start=note_on_time,
                    end=time_sec
                )
                instrument.notes.append(note)

            # Если новый pitch != -1, значит начинается новая нота
            if pitch != -1:
                current_pitch = pitch
                note_on_time = time_sec
            else:
                current_pitch = -1

    # Если в самом конце последняя нота не закрыта
    final_time = len(y_pred) * frame_duration
    if current_pitch != -1:
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_pitch,
            start=note_on_time,
            end=final_time
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(out_midi_path)
    print(f"[INFO] Saved predicted MIDI to {out_midi_path}")

################################ FUNCTIONS FOR API ################################
async def run_extract_and_save_data(mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz):
    loop = asyncio.get_running_loop()
    await asyncio.sleep(10)
    with ProcessPoolExecutor() as executor:
        await loop.run_in_executor(
            executor,
            extract_and_save_data,
            mp3_vocals_root,
            lmd_aligned_vocals_root,
            match_scores_json,
            output_npz
        )

async def fit(request: FitRequest):
    model = train_baseline_model_from_npz(FEATURES_DIR, request.hyperparameters)
    MODEL_NAME = request.id
    joblib.dump(model, os.path.join(MODELS_DIR, str(MODEL_NAME) + ".pkl"))
    print(f"[INFO] Saved model id {request.id}  to {FEATURES_DIR}")

def predict_model(MODEL_NAME, file: UploadFile = File(...)):
    try:
        loaded_model = joblib.load(os.path.join(MODELS_DIR, MODEL_NAME))
        print(f"[INFO] {MODEL_NAME} loaded.")
    except FileNotFoundError:
        loaded_model = None
        print(f"[WARN] {MODEL_NAME} not found, please train/save first.")

    if loaded_model is not None:
        audio, sr_ = librosa.load(file.file, sr=SR)
        # 1. Предикт (кадры -> MIDI pitch / -1)
        y_pred = predict_pitch_sequence(loaded_model, audio, sr_)

        # 2. Создаём MIDI из y_pred
        out_midi_path = os.path.join(PREDICTIONS_DIR, f"prediction_{file.filename}.mid")
        midi_from_prediction(
            y_pred=y_pred,
            hop_length=HOP_LENGTH,
            sr=SR,
            out_midi_path=out_midi_path
        )
        print("First 50 frames (MIDI or -1):", y_pred[:50])
        return(y_pred)