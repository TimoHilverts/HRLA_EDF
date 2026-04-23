from PostProcessing import PostProcessor
from BM_runMexperiments import U_z

files = [
    "temp_output/data/Mexperiment_M1_1773315684.385922.pickle",
    "temp_output/data/Mexperiment_M25_1773315726.2866004.pickle",
    "temp_output/data/Mexperiment_M50_1773315768.626037.pickle",
    "temp_output/data/Mexperiment_M75_1773315812.480637.pickle",
    "temp_output/data/Mexperiment_M100_1773315880.7412915.pickle",
]

PostProcessor.compare_M_table(files, a_idx=0, dpi=1, U=U_z)