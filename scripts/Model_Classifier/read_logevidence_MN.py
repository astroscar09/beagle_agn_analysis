import re
import glob
import numpy as np
import pandas as pd

def extract_logz_from_stats(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "Nested Sampling Global Log-Evidence" in line:
                # Extract values using regex
                match = re.search(r'([\-\d.Ee+]+)\s+\+/\-\s+([\d.Ee+]+)', line)
                if match:
                    logZ = float(match.group(1))
                    logZ_err = float(match.group(2))
                    return logZ, logZ_err
    return None, None  # If not found

model_files = {
    "Consistency Test": "results/consistency_test_run/1395_BEAGLE_MNstats.dat",
    "Final Run V2": "results/FINAL_RUN_V2/1395_BEAGLE_MNstats.dat",
    "CO Grid": "results/CO_Grids/1395_BEAGLE_MNstats.dat"
}

def main(model_files):
    
    results = []
    for label, file_path in model_files.items():
        logZ, logZ_err = extract_logz_from_stats(file_path)
        if logZ is not None:
            results.append((label, logZ, logZ_err, file_path))
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_logZ = results[0][1]
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=["Model", "logZ", "logZ_err", "File"])
    df["Delta_logZ"] = top_logZ - df["logZ"]
    df["Bayes Factor"] = np.exp(df["Delta_logZ"])
    
    return df

if __name__ == '__main__':

    
    main(model_files)