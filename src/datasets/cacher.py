from .preprocessor import Preprocessor

class Cacher:
    def __init__(self, *,
        train_indices, 
        val_indices, 
        test_indices, 
        data_path
    ):

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.data_path = data_path

    def make_cache(self):
        # loop through train, val and test indices
        # for each, make preprocessor process the scan
        # put them into their corresponding folder

        preprocessor = Preprocessor(self.data_path) # some args

        for i in self.train_indices:
            vol_arr, seg_arr = preprocessor.process(i)
            print(f"train: (vol shape: {vol_arr.shape}) (seg shape: {seg_arr.shape}")

        for i in self.val_indices:
            vol_arr, seg_arr = preprocessor.process(i)
            print(f"val: (vol shape: {vol_arr.shape}) (seg shape: {seg_arr.shape}")

        for i in self.test_indices:
            vol_arr, seg_arr = preprocessor.process(i)
            print(f"test: (vol shape: {vol_arr.shape}) (seg shape: {seg_arr.shape}")


