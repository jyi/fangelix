from typing import List, Tuple, Dict
import numpy as np

Chunk = int  # e.g. 2046
Block = np.ndarray  # e.g., array([0, 2046])
Sample = List[Block]  # [array([0, 2046]), array([0, 1])]

BitSeq = List[int]  # e.g., [0,1,1]
Proposal = List[BitSeq]

# An ebit represents the number of enabled bits.
# E.g., given [0,0,0,1], if an ebit is 2, then only the last two bits (0 and 1)
# are enabled.
# We also extend this concept to any number.
# For example, given [0,0,0,5], if an ebit is 2, then only the last two elements
# (0, 5) are enabled.
Ebits = int
EBitsSeq = List[Ebits]  # e.g., [1, 3, 3]

BlockEbits = Tuple[Block, Ebits]  # e.g., (array([ 7, 15, 14]), 11)

Cost = int

Location = Tuple[int, int, int, int]  # e.g., (12, 7, 12, 11)
Loc = Tuple[int, int, int, int]  # e.g., (12, 7, 12, 11)
LocGroup = List[Location]  # e.g., [(13, 7, 13, 11), (18, 15, 18, 19)

TestOut = str

Angel = Tuple
AngelicPath = Dict[Location, List[Angel]]

TraceFile = str
