"""Base Dataset Class"""

from typing import Any,Callable,Dict,Sequence,Tuple,Union
import torch 

SequenceOrTensor=Union[Sequence,torch.Tensor]

class BaseDataset(torch.utils.Dataset):
    """
    Base Dataset class makes and transforms the dataset

    Args:
        data: Tensors,Numpy Arrays ,PIL mages
        targets: Torch Tensors,Numpy Arrays
        transforms: function that takes data and return data
        target_transforsm: function that takes ad returns the same 

    """

    def __init__(self,data:SequenceOrTensor,targets:SequenceOrTensor,transform : Callable=None,target_tranform:Callable=None,) -> None :
        if len(data)!=len(targets):
            raise ValueError("Data and Targets must be of Equal Length")
            
    
        super().__init__()
        self.data=data 
        self.targets=targets
        self.transform=transform
        self.target_transform=target_tansform
    

    def __len__(self) -> int :
        """ Returns the length of the datasets """
        return len(self.data)
    

    def __getitem__(self,index:int) -> Tuple[Any,Any] :
        """
        Returns data and its targets after the appling the transformation


        Args:
            index (int): index of the data that needs to be returned

        Returns:
            Tuple[Any,Any]: The required data from all of the data 
        """
        datum,target=self.data[index],self.targets[index]


        if self.transform is not None:
            datum=self.transform(datum)
        
        if self.target_transform is not None:
            target=self.target_transforms

        
        return datum, target



def convert_strings_to_labels(strings: Sequence[str],mapping: Dict[str,int],length:int) -> torch.Tensor:
    """ 
    Convert sequence of N strings to a (N,length) narray with each string string wrapped with <S> and <E> tokens,
    and padded with the <P> token
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i,string in enumerate(strings):
        token=list(strings)
        tokens=["<S>",*tokens,"<E>"]

        for ii,token in enumerate(token):
            labels[i,ii]=mapping[token]

    return labels

def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )
    



