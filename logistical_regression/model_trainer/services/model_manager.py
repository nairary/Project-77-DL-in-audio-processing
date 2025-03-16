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
########################## BALANCING ################################
def balance_dataset(X: np.ndarray, Y: np.ndarray, silence_ratio: float = 0.3, 
                    min_class_samples: int = 50) -> tuple:
    print(f"[INFO] Original dataset shape: {X.shape}, {Y.shape}")
    
    # Get unique classes and their counts
    classes, counts = np.unique(Y, return_counts=True)
    class_indices = {cls: np.where(Y == cls)[0] for cls in classes}
    
    # Calculate target counts
    total_samples = len(Y)
    silence_samples = counts[0] if 0 in classes else 0
    non_silence_samples = total_samples - silence_samples
    
    # Determine target silence samples
    target_total = non_silence_samples / (1 - silence_ratio) if silence_ratio < 1.0 else total_samples
    target_silence = int(target_total * silence_ratio)
    
    # Ensure target_silence doesn't exceed current silence count
    target_silence = min(target_silence, silence_samples)
    
    # Calculate target counts for non-silence classes
    # Balance them while ensuring minimum samples
    non_silence_classes = [c for c in classes if c != 0]
    
    # Initialize array to collect balanced data
    X_balanced = []
    Y_balanced = []
    
    # Handle silence class (0)
    if 0 in classes:
        # Randomly select target_silence samples from silence class
        silence_indices = class_indices[0]
        selected_silence = np.random.choice(silence_indices, target_silence, replace=False)
        X_balanced.append(X[selected_silence])
        Y_balanced.append(Y[selected_silence])
        print(f"[INFO] Silence class reduced from {silence_samples} to {target_silence} samples")
    
    # Handle non-silence classes
    for cls in non_silence_classes:
        cls_indices = class_indices[cls]
        cls_count = len(cls_indices)
        
        # Skip empty classes
        if cls_count == 0:
            continue
        
            
        # If class has fewer than min_class_samples, use all samples
        if cls_count <= min_class_samples:
            selected_indices = cls_indices
        else:
            # Otherwise use min_class_samples or more
            target_cls_count = min(cls_count, min_class_samples * 5)  # Cap at 5x the minimum
            selected_indices = np.random.choice(cls_indices, target_cls_count, replace=False)
        
        X_balanced.append(X[selected_indices])
        Y_balanced.append(Y[selected_indices])
        print(f"[INFO] Class {cls} adjusted from {cls_count} to {len(selected_indices)} samples")
    
    # Combine all selected samples
    X_balanced = np.vstack(X_balanced)
    Y_balanced = np.concatenate(Y_balanced)
    
    # Shuffle the combined dataset
    shuffle_idx = np.random.permutation(len(Y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    Y_balanced = Y_balanced[shuffle_idx]
    
    print(f"[INFO] Balanced dataset shape: {X_balanced.shape}")
    
    # Print class distribution summary
    balanced_classes, balanced_counts = np.unique(Y_balanced, return_counts=True)
    print(f"[INFO] Top 5 classes in balanced data:")
    top_classes = np.argsort(balanced_counts)[-5:]
    for i in reversed(top_classes):
        cls = balanced_classes[i]
        count = balanced_counts[i]
        print(f"    Class {cls}: {count} samples ({count/len(Y_balanced)*100:.2f}%)")
    
    return X_balanced, Y_balanced

def augment_minority_classes(X: np.ndarray, Y: np.ndarray, 
                            min_samples_per_class: int = 100,
                            augmentation_factor: float = 1.2) -> tuple:
    # Determine classes that need augmentation
    classes, counts = np.unique(Y, return_counts=True)
    class_indices = {cls: np.where(Y == cls)[0] for cls in classes}
    
    X_aug_list = [X]
    Y_aug_list = [Y]
    
    # Skip silence class (0) for augmentation
    for cls in classes:
        if cls == 0:  # Skip silence class
            continue
            
        cls_count = counts[np.where(classes == cls)[0][0]]
        
        if cls_count < min_samples_per_class and cls_count > 0:
            # How many samples to add
            samples_to_add = min(min_samples_per_class - cls_count, cls_count * 2)
            
            # Get original samples for this class
            orig_indices = class_indices[cls]
            
            # Sample with replacement if we need more than we have
            sample_indices = np.random.choice(orig_indices, samples_to_add, replace=True)
            samples_to_augment = X[sample_indices]
            
            # Apply random noise augmentation
            noise_level = np.std(samples_to_augment) * augmentation_factor * 0.1
            noise = np.random.normal(0, noise_level, samples_to_augment.shape)
            augmented_samples = samples_to_augment + noise
            
            # Add augmented samples to our lists
            X_aug_list.append(augmented_samples)
            Y_aug_list.append(np.full(samples_to_add, cls))
            
            print(f"[INFO] Augmented class {cls} with {samples_to_add} new samples")
    
    # Combine original and augmented data
    X_augmented = np.vstack(X_aug_list)
    Y_augmented = np.concatenate(Y_aug_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(Y_augmented))
    X_augmented = X_augmented[shuffle_idx]
    Y_augmented = Y_augmented[shuffle_idx]
    
    print(f"[INFO] After augmentation: {X_augmented.shape}")
    return X_augmented, Y_augmented

class WeightedBatchSampler:
    """
    Sample mini-batches with class balancing for SGDClassifier training
    """
    def __init__(self, y, batch_size, class_weights=None):
        self.y = np.array(y)
        self.batch_size = batch_size
        self.classes, self.counts = np.unique(y, return_counts=True)
        self.n_samples = len(y)
        
        # Compute class weights if not provided
        if class_weights is None:
            # Inverse frequency weighting
            self.class_weights = 1.0 / (self.counts + 1)
            # Normalize
            self.class_weights = self.class_weights / np.sum(self.class_weights) * len(self.classes)
        else:
            self.class_weights = class_weights
        
        # Map classes to sample indices
        self.class_indices = {}
        for cls in self.classes:
            self.class_indices[cls] = np.where(self.y == cls)[0]
    
    def sample_batch(self):
        """Sample a balanced batch of indices"""
        batch_indices = []
        
        # Determine how many samples to take from each class
        # Inversely proportional to class frequency, with minimum samples for rare classes
        total_weight = np.sum([self.class_weights[i] for i, cls in enumerate(self.classes)])
        class_samples = {cls: max(1, int(self.batch_size * self.class_weights[i] / total_weight))
                        for i, cls in enumerate(self.classes)}
        
        # Adjust to match batch size
        total_selected = sum(class_samples.values())
        if total_selected < self.batch_size:
            # If under batch size, add samples from larger classes
            diff = self.batch_size - total_selected
            for cls in sorted(self.classes, key=lambda c: self.counts[np.where(self.classes == c)[0][0]], reverse=True):
                if diff <= 0:
                    break
                add_samples = min(diff, len(self.class_indices[cls]) - class_samples[cls])
                if add_samples > 0:
                    class_samples[cls] += add_samples
                    diff -= add_samples
        
        # Sample from each class
        for cls, n_samples in class_samples.items():
            if n_samples <= 0 or len(self.class_indices[cls]) == 0:
                continue
                
            # Sample with replacement if needed
            replacement = n_samples > len(self.class_indices[cls])
            sampled = np.random.choice(self.class_indices[cls], size=n_samples, replace=replacement)
            batch_indices.extend(sampled)
        
        # Final shuffle
        np.random.shuffle(batch_indices)
        return batch_indices[:self.batch_size]


################################ FIT ################################

def sgd_train_iterative(X: np.ndarray, y: np.ndarray, n_epochs: int = 10, 
                       batch_size: int = 1000, hyperparameters: Dict[str, Any] = {}) -> SGDClassifier:
    """
    Iteratively trains SGDClassifier on X, y with balanced mini-batches.
    
    Parameters:
      X, y        - feature matrix and labels (numpy arrays)
      n_epochs    - number of epochs (full passes over the data)
      batch_size  - mini-batch size
      hyperparameters - dictionary of model hyperparameters
    
    Returns trained SGDClassifier.
    """
    # Apply data balancing if requested
    if hyperparameters.get("balance_dataset", True):
        silence_ratio = hyperparameters.get("silence_ratio", 0.3)
        min_class_samples = hyperparameters.get("min_class_samples", 1000)
        X, y = balance_dataset(X, y, silence_ratio, min_class_samples)
    
    # Apply augmentation if requested
    if hyperparameters.get("augment_minority", True):
        min_samples = hyperparameters.get("min_samples_per_class", 50)
        aug_factor = hyperparameters.get("augmentation_factor", 1.2)
        X, y = augment_minority_classes(X, y, min_samples, aug_factor)

    classes_ = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes_, y=y)
    class_weight_dict = {cls: weight for cls, weight in zip(classes_, class_weights)}
    print(f"[INFO] Class weights: {class_weight_dict}")

    # Create model with hyperparameters
    clf = SGDClassifier(
        loss=hyperparameters.get("loss", "log_loss"),
        penalty=hyperparameters.get("penalty", "l2"),
        alpha=hyperparameters.get("alpha", 0.0001),
        max_iter=2,
        class_weight=class_weight_dict,
        warm_start=True,
        shuffle=False,
        n_jobs=hyperparameters.get("n_jobs", -1)
    )
    
    # Initialize with a small batch
    first_batch_size = min(batch_size, X.shape[0])
    X_init = X[:first_batch_size]
    y_init = y[:first_batch_size]
    clf.partial_fit(X_init, y_init, classes=classes_)

    # Use weighted batch sampler if requested
    use_weighted_sampler = hyperparameters.get("use_weighted_sampler", True)
    if use_weighted_sampler:
        batch_sampler = WeightedBatchSampler(y, batch_size)
    
    n_samples = X.shape[0]
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        
        batch_count = 0
        
        if use_weighted_sampler:
            # Training with weighted batch sampler
            n_batches = n_samples // batch_size
            for _ in range(n_batches):
                batch_indices = batch_sampler.sample_batch()
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                clf.partial_fit(X_batch, y_batch)
                batch_count += 1
        else:
            # Regular training with shuffled indices
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                clf.partial_fit(X_batch, y_batch)
                batch_count += 1
        
        print(f"Epoch {epoch+1} complete. Processed {batch_count} mini-batches.")

    return clf

def train_baseline_model_from_npz(input_npz: str, hyperparameters: Dict[str, Any]) -> SGDClassifier:
    """
    1) Loads X, Y from .npz
    2) Applies class balancing techniques
    3) Trains a classifier with balanced batches
    4) Returns the trained model
    """
    npz_files = [f for f in os.listdir(input_npz) if f.endswith('.npz')]
    first_npz_file = os.path.join(input_npz, npz_files[0])
    print(f"[INFO] Loading dataset from {first_npz_file}")
    
    data = np.load(first_npz_file)
    X_all = data["X"]
    Y_all = data["Y"]
    print(f"[INFO] Loaded dataset: X={X_all.shape}, Y={Y_all.shape}")
    
    # Get initial class distribution before balancing
    classes, counts = np.unique(Y_all, return_counts=True)
    print(f"[INFO] Initial class distribution: {len(classes)} classes")
    print(f"[INFO] Silence class (0) count: {counts[0]} ({counts[0]/len(Y_all)*100:.2f}%)")
    print(f"[INFO] Non-silence samples: {len(Y_all) - counts[0]} ({(len(Y_all) - counts[0])/len(Y_all)*100:.2f}%)")
    
    # Extract hyperparameters with defaults if not provided
    n_epochs = hyperparameters.get("n_epochs", 10)
    batch_size = hyperparameters.get("batch_size", 2000)
    
    # Train model with our improved training function
    clf = sgd_train_iterative(
        X_all, 
        Y_all, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        hyperparameters=hyperparameters
    )
    
    # Return the trained classifier
    print(f"[INFO] Model training complete")
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