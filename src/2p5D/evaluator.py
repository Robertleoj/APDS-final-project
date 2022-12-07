
import torch




class Evaluator:
    def __init__(self, config, net, device='cpu'):

        self.config = config
        self.net = net
        self.eval_batch_size = config['eval_batch_size']
        self.device = device
        

    def evaluate(self, vol):
        predictions = []

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())

        for batch in self.__make_batches(vol):
            print(f"{batch.shape=}")
            with torch.no_grad():
                out = self.net(batch)
                pred = out.argmax(1)
                predictions.append(pred.cpu())

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())
        
        return torch.concat(predictions, dim=0)

        
    def __make_batches(self, vol):

        all_slices = []

        vol_depth = vol.shape[0]

        for i in range(1, vol_depth - 1):
            all_slices.append(vol[i - 1: i + 2, :, :].unsqueeze(0))

        
        n_slices = len(all_slices)

        idx = 0
        while idx < n_slices:
            
            end_idx = min(idx + self.eval_batch_size, n_slices - 1)

            batch = torch.concat(all_slices[idx: end_idx + 1], dim=0)

            yield batch.to(self.device)

            idx = end_idx + 1
        
    
    def __make_empty(self):
        either_side = self.config['n_slices'] // 2
        empty = torch.zeros((either_side, 512, 512), dtype=torch.long)
        
        return empty

