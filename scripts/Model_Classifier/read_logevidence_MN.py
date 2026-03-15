import re
import glob
import numpy as np
import pandas as pd
from plot_best_model import plot_model_comparison

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
    
    "AGN_SF: NO": "final_ghz2_container/ReRuns/AGNMup100_NO_Grids_Rerun_MNstats.dat", 
    "AGN+SF: Mup100+CO": "Final_Runs/AGN_Mup100_Nebular_CO_Grids_w_logU_anshbranch_13_fixed_pl_MNstats.dat",
    "AGN+SF: Mup100 Default": 'Final_Runs/AGN_Mup100_logU_anshbranch_13_fixed_pl_MNstats.dat',
    "AGN+SF: Mup100+n4": "Final_Runs/AGN_Mup100_default_n4_anshbranch_13_fixed_pl_MNstats.dat",

    "AGN+SF: Mup100+CO+ AGN logU_w_oiii": "Final_Runs/AGN_Mup100_logU_CO_w_oiii_MNstats.dat",
    
    "AGN+SF: Mup300": "Final_Runs/AGN_Mup300_default_anshbranch_13_fixed_pl_MNstats.dat",
    "AGN+SF: Mup300+CO": "Final_Runs/AGN_Mup300_Nebular_CO_Grids_anshbranch_13_fixed_pl_MNstats.dat",
    "AGN+SF: Mup300+CO": "Final_Runs/AGN_Mup300_Nebular_CO_Grids_w_logU_anshbranch_13_fixed_pl_MNstats.dat",
    
    "SF Mup100": "Final_Runs/SF_Mup100_Default_anshbranch_13_fixed_pl_MNstats.dat",
    "SF Mup300": "Final_Runs/SF_Mup300_Default_anshbranch_13_fixed_pl_MNstats.dat",
    "SF Mup100 + Nebular CO": "Final_Runs/SF_Mup100_Nebular_CO_anshbranch_13_fixed_pl_MNstats.dat",
    "SF Mup300 + Nebular CO": "Final_Runs/SF_Mup300_Nebular_CO_anshbranch_13_fixed_pl_MNstats.dat",
    "SF Mup100 + n4": "Final_Runs/SF_Mup100_Default_n4_anshbranch_13_fixed_pl_MNstats.dat"
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

    
    df = main(model_files)
    plot_model_comparison(df)