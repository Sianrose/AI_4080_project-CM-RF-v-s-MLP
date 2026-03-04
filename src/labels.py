"""
Label mapping for Sign Language MNIST.

The dataset has 24 classes: letters A–Z excluding J and Z (those signs need motion).
Labels in the CSV are 0–23. This module maps label index → letter for display and evaluation.
"""

# 24 letters in order: A,B,C,D,E,F,G,H,I,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y
# Index 0 = A, 1 = B, ... 8 = I, 9 = K (J skipped), ... 23 = Y (Z skipped)
LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
]

NUM_CLASSES = len(LABELS)  # 24


def label_to_letter(label: int) -> str:
    """
    Convert a numeric label (0–23) to the corresponding letter.

    Args:
        label: Integer label from the dataset (0 to 23).

    Returns:
        Single letter string, e.g. "A", "K".

    Raises:
        IndexError: If label is not in 0–23.
    """
    if not 0 <= label < NUM_CLASSES:
        raise IndexError(f"Label must be 0–{NUM_CLASSES - 1}, got {label}")
    return LABELS[label]


def letter_to_label(letter: str) -> int:
    """
    Convert a letter to the numeric label (0–23).
    Useful for the app or custom inputs.

    Args:
        letter: Single letter, e.g. "A", "K". Case-insensitive.

    Returns:
        Integer label 0–23.

    Raises:
        ValueError: If letter is not one of the 24 valid letters.
    """
    letter = letter.upper()
    if letter not in LABELS:
        raise ValueError(f"Letter must be one of {LABELS}, got {letter}")
    return LABELS.index(letter)
