import functools

from processing.augmentation import ImageDataGenerator, random_crop, random_transform, samplewise, center_crop, \
    resize_image
# random_adapt_hist
import numpy as np
import os
from scipy.misc import imread
from tools.metrics import f1 as f1_score
from keras.utils.np_utils import to_categorical
import ast


def k_hot_encoding(x, num_classes=2):
    x_khe = np.zeros((num_classes), dtype=np.float32)
    x_khe[x] = 1.0
    return x_khe


class Dataset(object):
    # __metaclass__ = abc.ABCMeta

    NAME = 'dataset'

    # def __init(self, data_dir, data_type):
    #     # this function should be overriden
    #     pass

    # @abc.abstractmethod
    def load(self, data_dir, data_type):
        """Method documentation"""

    #
    # def id_to_category(self, primary_id):
    #     return self.CATEGORIES[primary_id]

    # @abc.abstractmethod
    def sample_generator(self):
        """Method documentation"""

class SlicingGenerators():
    def __init__(self, gen):
        self.gen = gen

    def __next__(self):
        x1, x2 = self.gen.__next__()
        print(x2[1])
        return x1[0], x2[0]



class Karolinska(Dataset):
    NAME = "Karolinska_Data"

    def __init__(self, config):
        self.mask_dir = config['main']['Karolinska_mask_dir']
        self.image_dir = config['main']['Karolinska_image_dir']
        self.root_dir = config['main']['root_dir']
        self.image_files = [file for file in os.listdir(os.path.join(self.root_dir, self.image_dir))]
        self.mask_files = [file for file in os.listdir(os.path.join(self.root_dir, self.mask_dir)) if
                           file.endswith('.png')]
        self.mask_files.sort()
        self.image_files.sort()
        self.nc = 3
        self.sample_number = len(self.image_files)
        self.class_names = 'Background', \
                           'benign',\
                            'cancerous'

        self.config = config
        self.dw = self.config["training"]["dw"]
        self.dh = self.config["training"]["dh"]
        self.input_shape = (self.config["training"]["dw"],
                            self.config["training"]["dh"], 3)
        self.class_weights = None

    def load_images(self):
        images = []
        for index in range(len(self.image_files)):
            temp_image = imread(os.path.join(self.root_dir, self.image_dir, self.image_files[index]))
            # temp_image = np.expand_dims(temp_image, axis=0)
            images.append(temp_image)
        # images=np.vstack(images)
        return images

    def load_masks(self):
        masks = []
        for index in range(len(self.mask_files)):
            temp_mask = imread(os.path.join(self.root_dir, self.mask_dir, self.mask_files[index]))
            # temp_mask=np.expand_dims(temp_mask,axis=0)
            #print(temp_mask.shape)
            temp_mask = np.expand_dims(temp_mask, axis=2)
            temp_mask = to_categorical(temp_mask, self.nc).reshape(temp_mask.shape[0], temp_mask.shape[1], self.nc)
            masks.append(temp_mask)
            # masks=np.vstack(masks)
            #print(temp_mask.shape)

        return masks

    def load_data(self):
        print(len(self.mask_files))
        print(len(self.image_files))
        return self.load_images(), self.load_masks()

    def create_generator(self):
        (images, masks) = self.load_data()
        # we create two instances with the same arguments
        # // featurewise_center = False,
        # // samplewise_center = False,
        # // featurewise_std_normalization = False,
        # // samplewise_std_normalization = False,
        # // featurewise_standardize_axis = None,
        # // samplewise_standardize_axis = None,
        # // zca_whitening = False,
        # // rotation_range = 0.,
        # // width_shift_range = 0.,
        # // height_shift_range = 0.,
        # // shear_range = 0.,
        # // zoom_range = 0.,
        # // channel_shift_range = 0.,
        # // fill_mode = 'nearest',
        # // cval = 0.,
        # // horizontal_flip = False,
        # // vertical_flip = False,
        # // rescale = None,
        # // dim_ordering = K.image_dim_ordering(),
        # // seed = None,
        # // verbose = 1):

        augment_params = self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)
        #
        # image_datagen.set_pipeline([random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        image_datagen.set_pipeline([random_transform])
        mask_datagen.set_pipeline([random_transform])

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        image_datagen.fit(images)
        mask_datagen.fit(masks)

        temp = np.ones(self.sample_number)
        ###########################

        #############

        image_generator = image_datagen.flow_from_list(images, temp, batch_size=self.config["training"]["batch_size"])
        mask_generator = mask_datagen.flow_from_list(masks,temp, batch_size=self.config["training"]["batch_size"])

        new_gen = image_generator + mask_generator

        gen = SlicingGenerators(new_gen)

        return gen  # image_generator, mask_generator

    def create_generator_with_validation(self):
        from sklearn.model_selection import train_test_split

        (images, masks) = self.load_data()

        augment_params = self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.set_pipeline([random_transform])
        mask_datagen.set_pipeline([random_transform])

        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)

        # image_datagen.set_pipeline([samplewise, random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)

        image_test = image_test

        mask_train = [masks[i] for i in index_train]
        mask_test = [masks[i] for i in index_test]

        image_datagen.fit(image_train)
        mask_datagen.fit(mask_train)

        temp = np.zeros(len(image_train))


        image_generator = image_datagen.flow_from_list(image_train, temp,
                                                       batch_size=self.config["training"]["batch_size"])
        mask_generator = mask_datagen.flow_from_list(mask_train, temp, batch_size=self.config["training"]["batch_size"])

        new_gen = image_generator + mask_generator

        gen = SlicingGenerators(new_gen)

        return gen, (batch_resize(image_test, (self.config["training"]["dw"],
                                               self.config["training"]["dh"]), rescale=1 / 255.0),
                     batch_resize(mask_test, (self.config["training"]["dw"],
                                              self.config["training"]["dh"])))

    def load_validation_data(self):
        from sklearn.model_selection import train_test_split
        (images, masks) = self.load_data()
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)
        masks_val = []
        for i in range(len(image_test)):
            mask_val = imread(os.path.join(self.root_dir, self.mask_dir, self.mask_files[index_test[i]]))
            masks_val.append(mask_val)
        image_val = image_test

        return image_val, masks_val


class Radboud (Dataset):
    NAME = "Radboud"

    def __init__(self, config):
        self.mask_dir = config['main']['Radboud_mask_dir']
        self.image_dir = config['main']['Radboud_image_dir']
        self.root_dir = config['main']['root_dir']
        self.image_files = [file for file in os.listdir(os.path.join(self.root_dir, self.image_dir)) ]
        self.mask_files = [file for file in os.listdir(os.path.join(self.root_dir, self.mask_dir)) ]
        self.mask_files.sort()
        self.image_files.sort()
        self.nc = 6
        self.sample_number = len(self.image_files)
        self.class_names = 'Background', \
                           'stroma', \
                           'healthy', \
                           'Gleason 3', \
                           'Gleason 4', \
                           'Gleason 5'
        self.config=config
        self.dw=self.config["training"]["ROI_dw"]
        self.dh=self.config["training"]["ROI_dh"]
        self.input_shape= (self.config["training"]["ROI_dw"],
                           self.config["training"]["ROI_dh"], 3)
        self.class_weights = None



    def load_images(self):
        images = []
        for index in range(len(self.image_files)):
            temp_image = imread(os.path.join(self.root_dir, self.image_dir, self.image_files[index]))
            # temp_image = np.expand_dims(temp_image, axis=0)
            images.append(temp_image)
        # images=np.vstack(images)
        return images

    def load_masks(self):
        masks = []
        for index in range(len(self.mask_files)):
            temp_mask = imread(os.path.join(self.root_dir, self.mask_dir, self.mask_files[index]))
            # temp_mask=np.expand_dims(temp_mask,axis=0)
            temp_mask = np.expand_dims(temp_mask, axis=2)
            temp_mask = to_categorical(temp_mask, self.nc).reshape(temp_mask.shape[0], temp_mask.shape[1], self.nc)
            masks.append(temp_mask)
        # masks=np.vstack(masks)


        return masks

    def load_data(self):
        print(len(self.mask_files))
        print(len(self.image_files))
        return self.load_images(), self.load_masks()

    def create_generator(self):
        (images, masks) = self.load_data()
        # we create two instances with the same arguments
        # // featurewise_center = False,
        # // samplewise_center = False,
        # // featurewise_std_normalization = False,
        # // samplewise_std_normalization = False,
        # // featurewise_standardize_axis = None,
        # // samplewise_standardize_axis = None,
        # // zca_whitening = False,
        # // rotation_range = 0.,
        # // width_shift_range = 0.,
        # // height_shift_range = 0.,
        # // shear_range = 0.,
        # // zoom_range = 0.,
        # // channel_shift_range = 0.,
        # // fill_mode = 'nearest',
        # // cval = 0.,
        # // horizontal_flip = False,
        # // vertical_flip = False,
        # // rescale = None,
        # // dim_ordering = K.image_dim_ordering(),
        # // seed = None,
        # // verbose = 1):

        augment_params= self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)
        #
        # image_datagen.set_pipeline([random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        image_datagen.set_pipeline([random_transform])
        mask_datagen.set_pipeline([random_transform])

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        image_datagen.fit(images)
        mask_datagen.fit(masks)

        temp = np.ones(self.sample_number)
        image_generator = image_datagen.flow_from_list(images, temp, batch_size=self.config["training"]["batch_size"])
        mask_generator = mask_datagen.flow_from_list(masks, temp, batch_size=self.config["training"]["batch_size"])

        new_gen = image_generator + mask_generator
        gen = SlicingGenerators(new_gen)

        return gen  # image_generator, mask_generator

    def create_generator_with_validation(self):
        from sklearn.model_selection import train_test_split

        (images, masks) = self.load_data()

        augment_params= self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.set_pipeline([random_transform])
        mask_datagen.set_pipeline([random_transform])


        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)

        # image_datagen.set_pipeline([samplewise, random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)

        image_test=image_test

        mask_train = [masks[i] for i in index_train]
        mask_test = [masks[i] for i in index_test]

        image_datagen.fit(image_train)
        mask_datagen.fit(mask_train)

        temp = np.ones(len(image_train))
        image_generator = image_datagen.flow_from_list(image_train, temp, batch_size=self.config["training"]["batch_size"])
        mask_generator = mask_datagen.flow_from_list(mask_train, temp, batch_size=self.config["training"]["batch_size"])

        new_gen = image_generator + mask_generator
        gen = SlicingGenerators(new_gen)

        return gen, (batch_resize(image_test, (self.config["training"]["ROI_dw"],
                           self.config["training"]["ROI_dh"]),rescale=1/255.0),
                     batch_resize(mask_test, (self.config["training"]["ROI_dw"],
                           self.config["training"]["ROI_dh"])))

    def load_validation_data(self):
        from sklearn.model_selection import train_test_split
        (images, masks) = self.load_data()
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)
        masks_val=[]
        for i in range(len(image_test)):

            mask_val = imread(os.path.join(self.root_dir, self.mask_dir, self.mask_files[index_test[i]]))
            masks_val.append(mask_val)
        image_val = image_test

        return image_val, masks_val

def batch_resize(X, target,rescale=1.0):
    X_size = X[0].shape
    if len(X_size) == 3:
        new_images = np.zeros((len(X), target[1], target[0], X_size[2]))
    else:
        new_images = np.zeros((len(X), target[1], target[0]))
    for i in range(len(X)):
        new_images[i] = resize_image(X[i], target)*rescale
    return new_images



class SlicingGenerators_2():
    def __init__(self, gen,gen2):
        self.gen = gen
        self.gen2 = gen2

    def __next__(self):
        x1, y1 = self.gen.__next__()
        x2, y2 = self.gen2.__next__()
        return [x1[0], x2[0]],[y1[0],y2[0]]

class Karolinska_Radboud(Dataset):
    NAME = "Karolinska_Radboud"
    def __init__(self, config):
        self.root_dir = config['main']['root_dir']

        self.Radboud = Radboud(config)
        self.Karolinska = Karolinska(config)
        self.nc = 6
        self.config = config
        self.dw = self.config["training"]["ROI_dw"]
        self.dh = self.config["training"]["ROI_dh"]

    def create_generator(self):
        (Radboud_images, Radboud_masks) = self.Radboud.load_data()
        (Karolinska_images, Karolinska_masks) = self.Karolinska.load_data()



        augment_params = self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        Radboud_image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        Radboud_mask_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        Karolinska_image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        Karolinska_mask_datagen = ImageDataGenerator(**data_gen_args)

        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)
        #
        # image_datagen.set_pipeline([random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        Radboud_image_datagen.set_pipeline([random_transform])
        Radboud_mask_datagen.set_pipeline([random_transform])
        Karolinska_image_datagen.set_pipeline([random_transform])
        Karolinska_mask_datagen.set_pipeline([random_transform])
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        Radboud_image_datagen.fit(Radboud_images)
        Radboud_mask_datagen.fit(Radboud_masks)
        Karolinska_image_datagen.fit(Karolinska_images)
        Karolinska_mask_datagen.fit(Karolinska_masks)
        ###load labels




        new_gen = Radboud_image_datagen + Radboud_mask_datagen
        new_gen_2 = Karolinska_image_datagen + Karolinska_mask_datagen
        gen = SlicingGenerators_2(new_gen,new_gen_2)

        return gen  # image_generator, mask_generator

    def create_generator_with_validation(self):
        from sklearn.model_selection import train_test_split

        (images, masks) = self.load_data()

        augment_params = self.config["augmentation"]
        data_gen_args = augment_params
        augment_params['rescale'] = 1 / 255.0
        image_datagen = ImageDataGenerator(**data_gen_args)

        data_gen_args = augment_params
        augment_params['rescale'] = None
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.set_pipeline([random_transform])
        mask_datagen.set_pipeline([random_transform])

        # image_datagen.config['random_crop_size'] = (300, 300)
        # mask_datagen.config['random_crop_size'] = (300, 300)
        # image_datagen.config['center_crop_size'] = (224, 224)
        # mask_datagen.config['center_crop_size'] = (224, 224)

        # image_datagen.set_pipeline([samplewise, random_crop, random_transform, center_crop])
        # mask_datagen.set_pipeline([random_crop, random_transform, center_crop])

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)

        image_test = image_test

        mask_train = [masks[i] for i in index_train]
        mask_test = [masks[i] for i in index_test]

        image_datagen.fit(image_train)
        mask_datagen.fit(mask_train)

        temp = np.ones(len(image_train))
        image_generator = image_datagen.flow_from_list(image_train, temp,
                                                       batch_size=self.config["training"]["batch_size"])
        mask_generator = mask_datagen.flow_from_list(mask_train, temp, batch_size=self.config["training"]["batch_size"])

        new_gen = image_generator + mask_generator
        gen = SlicingGenerators(new_gen)

        return gen, (batch_resize(image_test, (self.config["training"]["ROI_dw"],
                                               self.config["training"]["ROI_dh"]), rescale=1 / 255.0),
                     batch_resize(mask_test, (self.config["training"]["ROI_dw"],
                                              self.config["training"]["ROI_dh"])))

    def load_validation_data(self):
        from sklearn.model_selection import train_test_split
        (images, masks) = self.load_data()
        image_train, image_test, index_train, index_test = \
            train_test_split(images, np.arange(self.sample_number),
                             test_size=self.config["validation"]["split"],
                             random_state=14)
        masks_val = []
        for i in range(len(image_test)):
            mask_val = imread(os.path.join(self.root_dir, self.mask_dir, self.mask_files[index_test[i]]))
            masks_val.append(mask_val)
        image_val = image_test

        return image_val, masks_val
class SlicingGenerators_3():
    def __init__(self, gen,gen2):
        self.gen = gen
        self.gen2 = gen2

    def __next__(self):
        x1, y1 = self.gen.__next__()
        x2, y2 = self.gen2.__next__()
        return [x1[0], x2[0]],[y1[0],y2[1]]

