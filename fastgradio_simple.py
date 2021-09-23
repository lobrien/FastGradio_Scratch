# Use conda env pytorch-azureml
# Environment: export OMP_NUM_THREADS=1
from fastai.vision.all import *
from fastgradio import *
def run():
    path = untar_data(URLs.PETS)
    dls = ImageDataLoaders.from_name_re(path, get_image_files(path/'images'), pat='(.+)_\d+.jpg$', item_tfms=Resize(460), batch_tfms=aug_transforms(size=224, min_scale=0.755))
    learn = cnn_learner(dls, resnet50)
    learn.fine_tune(10)
    Demo(learn).launch()

if __name__ == '__main__':
    run()