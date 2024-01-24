from extract_dedup_data import extract_pruned_data

output_txt_path = "/workspace/CS762_Project/SemDeDup/code_alpaca_results/output.txt"
semdedup_pruning_tables_path = "/workspace/CS762_Project/SemDeDup/code_alpaca_results/save_location/dataframes"
sorted_clusters_path = "/workspace/CS762_Project/SemDeDup/code_alpaca_results/sorted_clusters"
eps = 0.9
# debug: supposed to be 1000...
num_clusters = 999

extract_pruned_data(sorted_clusters_path, semdedup_pruning_tables_path, eps, num_clusters, output_txt_path, retreive_kept_samples=True)