from fastai.vision.all import *
import rasterio
from PIL import ImageFilter

def get_multiband_img(chip_path):
    blue  = get_array(chip_path, 'B02')
    green = get_array(chip_path, 'B03')
    red   = get_array(chip_path, 'B04')
    infra = get_array(chip_path, 'B08')   
    stack = np.stack([blue, green, red, infra], axis = -1)##resulting size:(width,height,bands)
    return stack / stack.max()

def get_array(chip_path, band):
    fn = (chip_path / band).with_suffix('.tif')
    if fn.exists():
        arr = np.asarray(Image.open(fn))
        return arr #/ arr.max()
    else:
        return np.zeros((512, 512))
    
def get_mask(chip_path, label_path):
    fn = (label_path / chip_path.stem).with_suffix('.tif')
    return np.array(Image.open(fn))

class Chips:
    def __init__(self, src_path, df_src, debug = None):
        self.src_path = src_path
        self.df = self._get_df(df_src)
        self.paths = self._get_paths(debug)
       
    def _is_valid(self, chip):
        return self.df.loc[self.df['chip_id'] == chip.stem, 'is_valid'].item()
    
    def _get_paths(self, debug):
        if debug:
            self.df = self.df.sample(n=debug)
        chips = self.df['chip_id'].tolist()
        return L([self.src_path / chip for chip in chips])
    
    def _get_df(self, src):
        if isinstance(src, pd.DataFrame):
            return src
        elif isinstance(src, (Path, str)):
            return pd.read_csv(src)
        else:
            print('Can not load dataframe, should be pd.DataFrame or path to .csv')
        
    def get_paths(self):
        return self.names.map(lambda x: (self.src_path / x))
    
    def get_train_chips(self):
        self.train_idx = self.paths.argwhere(self._is_valid, negate = True)
        return self.paths[self.train_idx]
    
    def get_valid_chips(self):
        self.valid_idx = self.paths.argwhere(self._is_valid)
        return self.paths[self.valid_idx]
    
    def get_splits(self):
        if not hasattr(self, 'train_idx'):
            self.train_idx = self.paths.argwhere(self._is_valid, negate = True)
        if not hasattr(self, 'valid_idx'):
            self.valid_idx = self.paths.argwhere(self._is_valid)
        return [self.train_idx, self.valid_idx]

def show_b(b, ax = None, ctx = None, figsize = (15,15), title = None, alpha = 0.3, **kwargs):
    """
    Multiplies the mask into the yellow channel of the image with a factor of `alpha`.
    Adds the countours of the mask to the shown image
    """
    img, msk = b
    img = img.cpu().numpy()[:3].transpose(1,2,0)
    msk = msk.cpu().numpy().astype(np.uint8)
    contour = np.array(Image.fromarray(msk).filter(ImageFilter.FIND_EDGES))
    img[...,2] *= (1 - alpha * msk)
    img[np.where(contour)] = (0.7,0.1,0.3)
    
    ax = ifnone(ax,ctx)
    if ax is None:
        _,ax = plt.subplots(figsize = figsize)
    ax.imshow(img, **kwargs)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    return ax

@typedispatch
def show_batch(x, y, samples, ctxs=None, max_n=9, **kwargs):
    if ctxs is None: 
        ctxs = Inf.nones
    ctxs = [show_b(b, ctx = c, **kwargs) for b,c,_ in zip(samples,ctxs,range(max_n))]
    return ctxs