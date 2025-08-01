import numpy as np
import mne
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
import joblib # Using joblib as it was in previous scripts, though not strictly needed here

# --- 1. Load the list of subjects to evaluate ---
# We will create a personalized model for each subject in the test set.
try:
    test_subjects = np.load('test_subjects.npy')
    print(f"Loaded {len(test_subjects)} subjects for subject-specific evaluation.")
except FileNotFoundError:
    print("Error: 'test_subjects.npy' not found. Please run the training script first to create this file.")
    exit()


# A list to store the final accuracy for each personalized model
personalized_scores = []

print("\n--- Starting Subject-Specific Model Training and Evaluation ---")
# --- 2. Loop through each subject individually ---
for subject_id in test_subjects:
    print(f"\n--- Processing Subject {subject_id} ---")
    try:
        # --- A: Load and process all data for this single subject ---
        runs_lr = [4, 8, 12]
        runs_f = [6, 10, 14]
        fnames_lr = eegbci.load_data(subject_id, runs=runs_lr, verbose=False)
        fnames_f = eegbci.load_data(subject_id, runs=runs_f, verbose=False)
        raw_lr = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_lr])
        raw_f = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_f])

        def process_and_epoch(raw, event_id_map, event_id_labels):
            raw.filter(l_freq=8., h_freq=35., verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id=event_id_map, verbose=False)
            epochs = mne.Epochs(raw, events, event_id_labels, tmin=-0.5, tmax=3.5, preload=True,
                                baseline=None, picks='eeg', verbose=False)
            epochs.resample(160., verbose=False)
            return epochs

        epochs_lr = process_and_epoch(raw_lr, {'T1': 1, 'T2': 2}, {'left_fist': 1, 'right_fist': 2})
        epochs_f = process_and_epoch(raw_f, {'T2': 2}, {'both_feet': 2})
        
        all_epochs_for_subject = mne.concatenate_epochs([epochs_lr, epochs_f], verbose=False)
        
        data = all_epochs_for_subject.get_data()
        labels = all_epochs_for_subject.events[:, -1]

        # --- B: Split this subject's data into an 80% training set and a 20% test set ---
        # This simulates a calibration session (training) and then real-time use (testing).
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=42
        )
        print(f"  Data split: {len(train_labels)} trials for training, {len(test_labels)} for testing.")

        # --- C: Define the pipeline with your best hyperparameters ---
        csp = CSP(n_components=10, reg=None, log=True)
        svm = SVC(C=10, kernel='rbf', gamma='scale')
        personal_pipeline = Pipeline([('CSP', csp), ('Classifier', svm)])

        # --- D: Train the personalized model on this subject's training data ---
        print(f"  Training personalized model for Subject {subject_id}...")
        personal_pipeline.fit(train_data, train_labels)

        # --- E: Evaluate the personalized model on the rest of their data ---
        accuracy = personal_pipeline.score(test_data, test_labels)
        personalized_scores.append(accuracy)
        print(f"  ✅ Accuracy for Subject {subject_id} with personal model: {accuracy:.4f}")

    except Exception as e:
        print(f"  ❌ Skipping subject {subject_id} due to error: {e}")

# --- 3. Report Final Performance ---
print("\n--- Final Performance of Subject-Specific Models ---")
if personalized_scores:
    overall_mean = np.mean(personalized_scores)
    overall_std = np.std(personalized_scores)
    print(f"Mean accuracy across {len(personalized_scores)} personalized models: {overall_mean:.4f}")
    print(f"Standard deviation: {overall_std:.4f}")
