import pandas as pd


def calculate_ir_pr(file_path, descriptor, dmax_threshold):
    try:
        df = pd.read_excel(file_path)
        rare_samples = df[df['Dmax'] > dmax_threshold]
        n_rare = len(rare_samples)
        n_normal = len(df) - n_rare
        ir = n_rare / n_normal if n_normal != 0 else 0
        pr = n_rare / len(df) if len(df) != 0 else 0

        print(f"{descriptor} Descriptor:")
        print(f"Number of Rare Samples: {n_rare}")
        print(f"Number of Normal Samples: {n_normal}")
        print(f"Imbalance Ratio (IR): {ir:.3f}")
        print(f"Proportion of Rare Samples (PR): {pr * 100:.2f}%")

        return n_rare, n_normal, ir, pr

    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
    except Exception as e:
        print(f"An unknown error occurred: {e}")


file_path = ""
calculate_ir_pr(file_path, "T-Descriptor", 8)
calculate_ir_pr(file_path, "E-Descriptor", 2)
calculate_ir_pr(file_path, "G-Descriptor", 9)