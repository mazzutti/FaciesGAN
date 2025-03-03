import torchvision.transforms as transforms
from datasets.GenTIs import GslibDataset
from torch.utils.data import DataLoader

def get_dataloader(root:str):
    trans=transforms.Compose(
        [
            transforms.ToPILImage(),#transforms.ToTensor()接收的是PILImage格式数据,需要转换
            transforms.Resize(250),
            transforms.ToTensor(), # (H x W x C)->(C x H x W) in the range [0.0, 1.0]
            transforms.Normalize(0.5,1) 
        ]
    )
    dataset = GslibDataset(transform=trans,gslib_file=root)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    return loader