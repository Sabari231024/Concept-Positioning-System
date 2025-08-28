"""
This metric calculates the F1 score for a recommender system.
It compares the 5 predicted tags for each user against their 5 true tags, without considering order.

The solution DataFrame contains columns: UserID, Tag1, Tag2, Tag3, Tag4, Tag5.
The submission DataFrame contains columns: UserID, Prediction1, Prediction2, Prediction3, Prediction4, Prediction5.

All columns of the solution and submission dataframes are passed to the metric, except for the Usage column.
"""

import pandas as pd
import numpy as np
from typing import List


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Calculates the F1 score for a recommender system.

    Parameters:
    solution (pd.DataFrame): Ground truth data with UserID and Tag1-Tag5 columns.
    submission (pd.DataFrame): Predicted data with UserID and Prediction1-Prediction5 columns.
    row_id_column_name (str): Name of the column containing user IDs (should be 'UserID').

    Returns:
    float: F1 score, rounded to 2 decimal places.

    >>> import pandas as pd
    >>> row_id_column_name = "UserID"
    >>> solution = pd.DataFrame({
    ...     'UserID': [1, 2, 3],
    ...     'Tag1': [23, 56, 69],
    ...     'Tag2': [31, 33, 42],
    ...     'Tag3': [69, 214, 8],
    ...     'Tag4': [42, 123, 11],
    ...     'Tag5': [44, 121, 5]
    ... })
    >>> submission = pd.DataFrame({
    ...     'UserID': [1, 2, 3],
    ...     'Prediction1': [23, 56, 69],
    ...     'Prediction2': [31, 33, 42],
    ...     'Prediction3': [69, 214, 8],
    ...     'Prediction4': [42, 123, 11],
    ...     'Prediction5': [44, 121, 5]
    ... })
    >>> score(solution, submission, row_id_column_name)
    1.0
    '''

    expected_solution_columns = ['UserID', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']
    expected_submission_columns = ['UserID', 'Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Prediction5']

    if not all(col in solution.columns for col in expected_solution_columns):
        raise ParticipantVisibleError(f"Solution must contain columns: {', '.join(expected_solution_columns)}")

    if not all(col in submission.columns for col in expected_submission_columns):
        raise ParticipantVisibleError(f"Submission must contain columns: {', '.join(expected_submission_columns)}")

    f1_scores: List[float] = []

    for idx, row in solution.iterrows():
        true_set = set(row[['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']])
        pred_set = set(submission.loc[idx, ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Prediction5']])

        if len(pred_set) == 0:
            f1_scores.append(0)
        else:
            intersection = len(true_set.intersection(pred_set))
            precision = intersection / len(pred_set)
            recall = intersection / len(true_set)

            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)

    return round(np.mean(f1_scores), 2)