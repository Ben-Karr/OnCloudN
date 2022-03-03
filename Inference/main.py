import albumentations as A
from fastai.vision.all import *
from assets.chiputility import *
from loguru import logger
import typer
import skimage.morphology # remove small objects from prediction
import torch 
import gc # collect gpu memory
import os # to remove tmp folder

def get_chips(path):
    potential_chips = list(path.iterdir())
    chips_paths = L(chip for chip in potential_chips if chip.is_dir())
    return chips_paths.attrgot('name')

def save_probs(probs, i_batch, bs, chips, path, k):
    i = i_batch * bs
    for j in range(probs.shape[0]):
        chip_id = chips[i+j].stem
        #fn = (path / (chip_id + f'_{k}')).with_suffix('.npy')
        fn = (path / chip_id).with_suffix('.npy')
        
        if fn.exists():
            old_probs = np.load(fn)
            probs[j] += old_probs
        np.save(fn, probs[j])

def post_process(mask, cutoff = 0.5, min_object_size = 100.):
    label_mask = skimage.morphology.label(mask > cutoff)

    labels = set(label_mask.flatten())
    if len(labels) == 1: ## len == 1 means all mask or no mask, nothing to remove
        return mask
    labels.remove(0) ## remove background label, else all background becomes object
    prediction = np.zeros(shape = (512, 512), dtype = np.uint8)
    for l in labels:
        p = (label_mask == l).astype(np.uint8)
        if p.sum() >= min_object_size:
            prediction += p
    return prediction

def save_imgs(tmp_path, preds_path, n_models):
    for fn in tmp_path.iterdir():
        preds = np.load(fn)
        mask_arr = (preds / n_models) > 0.5 # loads sum (ensemble) of predicted probs for cloud
        
        mask_arr = post_process(mask_arr, min_object_size = 50) # remove small objects
        mask = Image.fromarray(mask_arr.astype(np.uint8))
                
        smth_mask = mask.filter(ImageFilter.ModeFilter(size = 10)) # smooth mask
        smth_mask.save((preds_path / fn.name).with_suffix('.tif'))

class OnlyVizAlbumentationsTransform(Transform):
    split_idx = 0
    def __init__(self, aug):
        self.aug = aug
    def encodes(self, x):
        if len(x.shape) > 2:
            non_viz_channels = x[...,3:]
            viz_aug = self.aug(image=x[...,:3].astype(np.float32))['image']
            return np.concatenate([viz_aug,non_viz_channels], axis = -1)
        else:
            return x

class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    def __init__(self, aug): 
        self.aug = aug
    def encodes(self, x):
        augs = []
        for img,mask in x:
            augs.append(tuple(self.aug(image=img, mask=mask).values()))
        return augs

class TransposeTransform(ItemTransform):
    def encodes(self, x):
        transposed = []
        for img in x:
            transposed.append(TensorImage(img[0].transpose(2,0,1)).float(),)
        return transposed

class FormatTransform(ItemTransform):
    def __init__(self, return_type):
        self.return_type = return_type
    def encodes(self, x):
        return self.return_type([TensorImage(x[0].permute(0,3,1,2)).float(), TensorMask(x[1]).long()])
def main():
    ## Setup paths
    logger.info('Setting up stuff')
    ROOT_DIRECTORY = Path('.')
    test_path = ROOT_DIRECTORY / 'data' / 'test_features'
    preds_path = ROOT_DIRECTORY / 'predictions'
    model_paths = [ROOT_DIRECTORY /'assets/model_old_split', 
                ROOT_DIRECTORY / 'assets/model_new_split',
                ROOT_DIRECTORY / 'assets/model_fp16',
                ]
    tmp_path = ROOT_DIRECTORY / 'tmp'

    ## Setup tfms
    logger.info('Setting up tfms to load learner')
    viz_augs_list = A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p = 0.5
        ),
        #A.Normalize(max_pixel_value = 1),
        A.RandomBrightnessContrast(),
    ])

    geom_augs_list = A.Compose([
        A.Flip(),
        #A.RandomCrop(440, 440),
        A.RandomGridShuffle(grid = (2,2), p = 0.3),
        A.CoarseDropout(mask_fill_value = 0),
        ])

    viz_augs_tfms = OnlyVizAlbumentationsTransform(viz_augs_list)
    geom_augs_tfms = SegmentationAlbumentationsTransform(geom_augs_list)
    transpose_tfm = TransposeTransform()
    format_tfm = FormatTransform(tuple)

    ## Make folder for predictions
    ## Should be build already
    logger.info('Create folder for predictions')
    preds_path.mkdir(exist_ok = True, parents=True)
    tmp_path.mkdir(exist_ok = True)

    ## Create DataFrame to fit learners transform pipeline
    logger.info('Create Data source')
    test_df = pd.DataFrame({'chip_id': get_chips(test_path)})
    chips = Chips(test_path, test_df)

    for k, model_path in enumerate(model_paths):
        ## Load pretrained learner with GPU
        logger.info(f'Load Learner {k+1} of {len(model_paths)}')
        learn = load_learner(model_path, cpu = False)
        learn.dls.loaders[0].before_batch = Pipeline(transpose_tfm)
        test_dl = learn.dls.test_dl(chips.paths, bs = 8)
        ## Get confidence of being cloud and save as array
        bs = test_dl.bs
        logger.info(f'Predict on {len(test_df)} chips with batch size of {bs}')
        with torch.no_grad():
            n_inst = 0
            for i,b in enumerate(test_dl):
                if (i % 50 == 0):
                    logger.info(f'Process batch nr: {i}')
                preds = learn.model(b).cpu()
                soft_preds = torch.nn.functional.softmax(preds, dim=1)
                prob_preds = soft_preds[:,1,...].numpy().astype(np.float16) #float16 to save memory
                save_probs(prob_preds, i, bs, chips.paths, tmp_path, k = k)

                n_inst += preds.shape[0]
        logger.info(f'Finished predicting {n_inst} probabilities with Learner {k+1}.')
        del(learn)
        del(test_dl)
        torch.cuda.empty_cache()
        gc.collect()
    logger.info(f'Ensemble predictions and save as .tif')
    save_imgs(tmp_path, preds_path, len(model_paths))
    logger.info('Clean up')
    os.system('rm -rf tmp')
    logger.info(f'Done')

if __name__ == "__main__":
    typer.run(main)
