

python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 14_out_clusters.pkl -sr 14_out_rep.pkl -ca hdbscan -cam l2 -cams None -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 >> 14_test_umap.txt;


python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 15_out_clusters.pkl -sr 15_out_rep.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 3 >> 15_test_umap.txt;


python3 -u baseline_plus_bow_on_traditional_clustering_w_metrics.py -sc 115_out_clusters.pkl -sr 115_out_rep.pkl -ca hdbscan -cam l2 -cams 1 -casm eom -caas 0 -dra umap -drnc 15 -drm euclidean -n 7 -t 0 >> 115_test_umap.txt;
