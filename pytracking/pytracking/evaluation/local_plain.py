from pytracking.evaluation.environment import EnvSettings

def local_env_settings():

    print("loading testing data")
    
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = "tracking_dataset_path/GOT10K/"  # got10k_test got10k_val
    settings.lasot_path = "tracking_dataset_path/lasot/" # lasot
    settings.nfs_path = "tracking_dataset_path/nfs_pytracking/" # nfs
    settings.otb_path = "tracking_dataset_path/OTB2015/" # otb
    settings.uav_path = "tracking_dataset_path/UAV123/" # uav
    settings.lasot_extension_subset_path = "tracking_dataset_path/lasotext/data/LaSOT_extension_subset/"
    settings.trackingnet_path = "/tracking_dataset_path/trackingnet" # trackingnet
    settings.avist_path = "tracking_dataset_path/AVisT/avist/"

    settings.network_path = "/pytracking_path/networks/"    # Where tracking networks are stored.
    settings.result_plot_path = '/pytracking_path/result_plots/'
    settings.results_path = '/pytracking_path/tracking_results/'    # Where to store tracking results

    return settings







