import pandas as pd
import numpy as np


def get_splits(age_at_exam, patient_ids, exam_ids, splits, min_age_valid=16, max_age_valid=85, seed=0):
    rng = np.random.RandomState(seed)
    if sum(splits) > 1.0:
        raise ValueError('splits should be sum to a number smaller than one.')
    n_exams = len(exam_ids)
    # Get patients
    patients = np.unique(patient_ids)
    n_patients = len(patients)
    # Create correspondence
    hash_exams = dict(zip(exam_ids, range(n_exams)))
    hash_patients = dict(zip(patients, range(n_patients)))
    inverse_hash_patients = dict(zip(range(n_patients), patients))
    # Get all exams for each patient
    patient_exams = [[] for _ in range(n_patients)]
    for exam_idx in range(n_exams):
        exam_id = exam_ids[exam_idx]
        patient_id = patient_ids[exam_idx]
        patient_idx = hash_patients[patient_id]
        patient_exams[patient_idx].append(exam_id)

    # Get the age at one of the exams for each patient
    ages, _, _ = np.unique(age_at_exam, return_inverse=True, return_counts=True)
    patient_idx_per_age = {a: [] for a in ages}
    patient_single_exam = np.zeros(n_patients, dtype=int)
    for patient_idx in range(n_patients):
        # Pick random exam id for the given patient
        # OBS:Another formulation that could make sense could be to always pick the first exam....
        id_exam = rng.choice(patient_exams[patient_idx])
        patient_single_exam[patient_idx] = id_exam
        exam_idx = hash_exams[id_exam]
        a = age_at_exam[exam_idx]
        patient_idx_per_age[a].append(patient_idx)

    # Get number of patient in each split
    n_splits = [int(np.floor(s * n_patients)) for s in splits]
    n_splits += [n_patients - sum(n_splits)]
    # Shuffle
    rng.shuffle(ages)  # Shuffle ages
    for a, patient_idx in patient_idx_per_age.items():  # Shuffle within the same age
        rng.shuffle(patient_idx)
    # Pick one id per age and build a list from that
    all_patient_idx = []
    stop = False
    # Pick ids within the given range first (which will probably be used for training, validation and test)
    while not stop:
        stop = True
        for a in ages[(ages >= min_age_valid) & (ages <= max_age_valid)]:
            if patient_idx_per_age[a]:
                patient_idx = patient_idx_per_age[a].pop()
                all_patient_idx.append(patient_idx)
                stop = False
    # Pick remaining ids last (which will probably be used only for training)
    stop = False
    while not stop:
        stop = True
        for a in ages[(ages < min_age_valid) | (ages > max_age_valid)]:
            if patient_idx_per_age[a]:
                patient_idx = patient_idx_per_age[a].pop()
                all_patient_idx.append(patient_idx)
                stop = False
    # Save ids
    patients_in_splits = [[] for n in n_splits]
    single_exam_in_split = [[] for n in n_splits]
    exams_in_splits = [[] for n in n_splits]
    for i, patient_idx in enumerate(all_patient_idx):
        last_n = 0
        for s, n in enumerate(np.cumsum(n_splits)):
            if last_n <= i < n:
                patients_in_splits[s].append(inverse_hash_patients[patient_idx])
                single_exam_in_split[s].append(patient_single_exam[patient_idx])
                exams_in_splits[s] += patient_exams[patient_idx]
                last_n = n

    return patients_in_splits, single_exam_in_split, exams_in_splits


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='Generate data summary for the age prediction problem')
    parser.add_argument('file',
                        help='csv file to read data from.')
    parser.add_argument('--exam_id_col', default='N_exame',
                        help='column in csv containing exam id')
    parser.add_argument('--age_col', default='Idade',
                        help='column in csv containing age')
    parser.add_argument('--patient_id_col', default='N_paciente_univoco',
                        help='column in csv containing patient id')
    parser.add_argument('--splits', default=[0.15, 0.05], nargs='*', type=float,
                        help='percentage of data in each split')
    parser.add_argument('--splits_names', default=['test', 'val', 'train'], nargs='*', type=str,
                        help='split names')
    parser.add_argument('--no_plot', action='store_true',
                        help='dont show plots')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Open csv file
    df = pd.read_csv(args.file, low_memory=False)
    # Remove duplicated rows
    df.drop_duplicates(args.exam_id_col, inplace=True)
    # Get ids from csv file
    exam_ids = np.array(df[args.exam_id_col], dtype=int)
    age_at_exam = np.array(df[args.age_col])
    patient_ids = np.array(df[args.patient_id_col], dtype=int)
    # define splits
    patients_in_splits, single_exam_in_split, exams_in_splits = get_splits(age_at_exam, patient_ids, exam_ids, args.splits)

    if not args.no_plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        n = len(args.splits) + 1
        fig, ax = plt.subplots(nrows=n)
        for i in range(n):
            age_single_exam = age_at_exam[np.isin(exam_ids, single_exam_in_split[i])]
            age = age_at_exam[np.isin(exam_ids, exams_in_splits[i])]
            sns.histplot(age, ax=ax[i], kde=False, bins=range(0, 130, 1))
            sns.histplot(age_single_exam, ax=ax[i], kde=False, bins=range(0, 130, 1))
        plt.show()

