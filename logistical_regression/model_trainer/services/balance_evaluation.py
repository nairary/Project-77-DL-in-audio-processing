def evaluate_pitch_model(model: SGDClassifier, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the pitch detection model with metrics specific to this task.
    
    Parameters:
    -----------
    model : SGDClassifier
        The trained model to evaluate
    X_test : np.ndarray
        Test feature matrix
    Y_test : np.ndarray
        True test labels
    
    Returns:
    --------
    Dict[str, Any]: Dictionary of evaluation metrics
    """
    # Get predictions
    Y_pred = model.predict(X_test)
    
    # Basic metrics
    correct = Y_pred == Y_test
    accuracy = np.mean(correct)
    
    # Separate silence and non-silence indices
    silence_idx = Y_test == 0
    non_silence_idx = ~silence_idx
    
    # Calculate silence and non-silence accuracy
    silence_accuracy = np.mean(correct[silence_idx]) if np.any(silence_idx) else 0
    non_silence_accuracy = np.mean(correct[non_silence_idx]) if np.any(non_silence_idx) else 0
    
    # One semitone accuracy (for non-silence only)
    # Where prediction is within ±1 semitone of the true pitch
    one_semitone_correct = np.zeros_like(correct, dtype=bool)
    if np.any(non_silence_idx):
        pitch_diff = np.abs(Y_pred[non_silence_idx] - Y_test[non_silence_idx])
        one_semitone_correct[non_silence_idx] = (pitch_diff <= 1) & (pitch_diff > 0)
    one_semitone_accuracy = np.mean(one_semitone_correct[non_silence_idx]) if np.any(non_silence_idx) else 0
    
    # Combined accuracy: either exactly correct or within one semitone
    combined_correct = correct | one_semitone_correct
    combined_accuracy = np.mean(combined_correct)
    non_silence_combined_accuracy = np.mean(combined_correct[non_silence_idx]) if np.any(non_silence_idx) else 0
    
    # Get class distribution in test set
    classes, counts = np.unique(Y_test, return_counts=True)
    class_distribution = {int(cls): int(count) for cls, count in zip(classes, counts)}
    
    # Collect metrics
    metrics = {
        "accuracy": float(accuracy),
        "silence_accuracy": float(silence_accuracy),
        "non_silence_accuracy": float(non_silence_accuracy),
        "one_semitone_accuracy": float(one_semitone_accuracy),
        "combined_accuracy": float(combined_accuracy),
        "non_silence_combined_accuracy": float(non_silence_combined_accuracy),
        "class_distribution": class_distribution
    }
    
    # Print summary
    print(f"[EVAL] Overall accuracy: {accuracy:.4f}")
    print(f"[EVAL] Silence accuracy: {silence_accuracy:.4f}")
    print(f"[EVAL] Non-silence accuracy: {non_silence_accuracy:.4f}")
    print(f"[EVAL] Within one semitone accuracy: {one_semitone_accuracy:.4f}")
    print(f"[EVAL] Combined (exact or ±1 semitone): {combined_accuracy:.4f}")
    
    return metrics

def extract_and_save_balanced_data(mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, 
                                  output_npz, collision_resolver='min',
                                  balance_params=None) -> None:
    """
    Enhanced version of extract_and_save_data that includes balancing options.
    
    Parameters:
    -----------
    mp3_vocals_root : str
        Path to vocals audio files
    lmd_aligned_vocals_root : str
        Path to aligned MIDI files
    match_scores_json : str
        Path to matching scores JSON file
    output_npz : str
        Path to save features and labels
    collision_resolver : str
        Strategy for resolving pitch collisions ('min' or 'max')
    balance_params : dict
        Parameters for balancing the dataset
    """
    # Set default balance parameters if not provided
    if balance_params is None:
        balance_params = {
            "enable_balancing": True,
            "silence_ratio": 0.3,
            "min_class_samples": 50,
            "enable_augmentation": True,
            "min_samples_per_class": 100
        }
    
    # Run original data extraction
    global MP3_VOCALS_DIR, MIDI_VOCALS_DIR, MATCH_SCORES_PATH, FEATURES_DIR
    
    MP3_VOCALS_DIR = mp3_vocals_root
    MIDI_VOCALS_DIR = lmd_aligned_vocals_root
    MATCH_SCORES_PATH = match_scores_json
    FEATURES_DIR = output_npz + ".npz"
    
    with open(MATCH_SCORES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Map MSD ID to best_md5
    best_md5_map = {}
    for msd_id, md5_scores in data.items():
        best_md5 = None
        best_score = float('-inf')
        for m, sc in md5_scores.items():
            if sc > best_score:
                best_score = sc
                best_md5 = m
        best_md5_map[msd_id] = best_md5
    
    # Get all MSD IDs
    msd_ids = []
    for d in os.listdir(MP3_VOCALS_DIR):
        if d.startswith("TR") and os.path.isdir(os.path.join(MP3_VOCALS_DIR, d)):
            msd_ids.append(d)
    
    X_all, Y_all = [], []
    
    # Extract features and labels
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
        
        # Generate labels
        labels = generate_labels_from_vocal_midi(vocals_wav, midi_path, collision_resolver)
        
        # Extract features
        audio, _ = librosa.load(vocals_wav, sr=SR)
        feats = extract_features(audio, SR)
        
        # Ensure matching lengths
        min_len = min(len(labels), feats.shape[0])
        labels = labels[:min_len]
        feats = feats[:min_len]
        
        # Convert pitch to class
        class_labels = [midi_to_class(p) for p in labels]
        
        X_all.append(feats)
        Y_all.append(class_labels)
        
        print(f"[OK] {msd_id}, frames={min_len}")
    
    if len(X_all) == 0:
        print("[WARN] No data found to save.")
        return
    
    # Concatenate all features and labels
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    
    # Apply balancing if enabled
    if balance_params["enable_balancing"]:
        print("[INFO] Applying dataset balancing...")
        X_all, Y_all = balance_dataset(
            X_all, Y_all, 
            silence_ratio=balance_params["silence_ratio"],
            min_class_samples=balance_params["min_class_samples"]
        )
        
        # Apply augmentation if enabled
        if balance_params["enable_augmentation"]:
            print("[INFO] Applying minority class augmentation...")
            X_all, Y_all = augment_minority_classes(
                X_all, Y_all,
                min_samples_per_class=balance_params["min_samples_per_class"]
            )
    
    # Save balanced dataset
    print(f"[INFO] Saving processed dataset: X={X_all.shape}, Y={Y_all.shape} to {FEATURES_DIR}")
    np.savez(FEATURES_DIR, X=X_all, Y=Y_all, allow_pickle=False)
    print("[INFO] Done saving dataset.")

    def merge_pitch_classes(Y: np.ndarray, strategy: str = 'octave') -> np.ndarray:
    """
    Merges pitch classes to reduce the number of classes.
    
    Parameters:
    -----------
    Y : np.ndarray
        Original labels
    strategy : str
        Merging strategy:
        - 'octave': Group pitches by octave (C1, C2, C3 all become C)
        - 'semitone3': Group every 3 semitones
        - 'semitone6': Group every 6 semitones (half octave)
    
    Returns:
    --------
    np.ndarray: Merged class labels
    """
    # Make a copy of the labels
    Y_merged = Y.copy()
    
    # Silence class (0) remains the same
    non_silence_mask = Y > 0
    
    if strategy == 'octave':
        # Convert class back to MIDI pitch (subtract 1)
        # Then mod 12 to get octave-invariant pitch class
        # Add 1 back to avoid conflict with silence class
        Y_merged[non_silence_mask] = ((Y_merged[non_silence_mask] - 1) % 12) + 1
        
    elif strategy == 'semitone3':
        # Group every 3 semitones (pitch classes)
        # First convert to MIDI pitch
        midi_pitch = Y_merged[non_silence_mask] - 1
        # Get modulo 12 pitch class
        pitch_class = midi_pitch % 12
        # Group into 4 groups of 3 semitones each
        grouped_class = pitch_class // 3
        # Convert back to class format
        Y_merged[non_silence_mask] = grouped_class + 1  # Add 1 to avoid conflict with silence
        
    elif strategy == 'semitone6':
        # Group every 6 semitones (half octave)
        # First convert to MIDI pitch
        midi_pitch = Y_merged[non_silence_mask] - 1
        # Get modulo 12 pitch class
        pitch_class = midi_pitch % 12
        # Group into 2 groups of 6 semitones each
        grouped_class = pitch_class // 6
        # Convert back to class format
        Y_merged[non_silence_mask] = grouped_class + 1  # Add 1 to avoid conflict with silence
    
    return Y_merged