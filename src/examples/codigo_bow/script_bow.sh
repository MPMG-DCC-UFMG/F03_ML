

python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 14_out_clusters_with_outlier.pkl -sr 14_out_rep_with_outlier.pkl -ca hdbscan -cam l2 -cams None -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 -rmo 0 >> 14_test_umap_with_outlier.txt;


python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 14_out_clusters_without_outlier.pkl -sr 14_out_rep_without_outlier.pkl -ca hdbscan -cam l2 -cams None -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 -rmo 1 >> 14_test_umap_without_outlier.txt;


python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 15_out_clusters_with_outliers.pkl -sr 15_out_rep_with_outliers.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 3 -rmo 0 >> 15_test_umap_with_outliers.txt;

python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 15_out_clusters_without_outliers.pkl -sr 15_out_rep_without_outliers.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 3 -rmo 1 >> 15_test_umap_without_outliers.txt;

python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 115_out_clusters_with_outliers.pkl -sr 115_out_rep_with_outliers.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 -rmo 0 >> 115_test_umap_with_outliers.txt;

python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 115_out_clusters_without_outliers.pkl -sr 115_out_rep_without_outliers.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 -rmo 1 >> 115_test_umap_without_outliers.txt;
