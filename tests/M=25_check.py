import numpy as np
from PostProcessing import PostProcessor
from BasicModel import U, dU

# load the experiment file
pp = PostProcessor(
    "temp_output/data/Mexperiment_M25_1773315726.2866004.pickle",
    U=U,
    dU=dU
)

# compute the statistics for this M
summary = pp.summarize_best_of_M(a_idx=0, dpi=500)

print(summary)