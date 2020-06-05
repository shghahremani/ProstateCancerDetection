
def select_dataset(config):
    global selected_dataset
    dataset_name=config['main']['dataset_name']

    if dataset_name == 'Karolinska_Radboud':
        from dataset.dataset import Karolinska_Radboud as selected_dataset
        return selected_dataset(config)
