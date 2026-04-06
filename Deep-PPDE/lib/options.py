import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

@dataclass
class BaseOption:
    pass
    
    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...



class Lookback(BaseOption):
    
    def __init__(self, idx_traded: List[int]=None):
        self.idx_traded = idx_traded # indices of traded assets. If None, then all assets are traded
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            lookback option payoff. Tensor of shape (batch_size,1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded],2) # (batch_size, N)
        else:
            basket = torch.sum(x,2) # (batch_size, N)
        payoff = torch.max(basket, 1)[0]-basket[:,-1] # (batch_size)
        return payoff.unsqueeze(1) # (batch_size, 1)


class Autocallable(BaseOption):
    
    def __init__(self, idx_traded: int, B: int, Q1: float, Q2: float, q: float, r: float, ts: torch.Tensor):
        """
        Autocallable option with 
        - two observation dates (T/3, 2T/3), 
        - premature payoffs Q1 and Q2
        - redemption payoff q*s
        """
        
        self.idx_traded = idx_traded # index of traded asset
        self.B = B # barrier
        self.Q1 = Q1
        self.Q2 = Q2
        self.q = q # redemption payoff
        self.r = r # risk-free rate
        self.ts = ts # timegrid
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            autocallable option payoff. Tensor of shape (batch_size,1)
        """
        id_t1 = len(self.ts)//3
        mask1 = x[:, id_t1, self.idx_traded]>=self.B
        id_t2 = 2*len(self.ts)//3
        mask2 = x[:, id_t2, self.idx_traded]>=self.B

        payoff = mask1 * self.Q1 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t1])) # we get the payoff Q1, and we put in a risk-less acount for the remaining time
        payoff += ~mask1 * mask2 * self.Q2 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t2]))
        payoff += ~mask1 * (~mask2) * self.q*x[:,-1,self.idx_traded]

        return payoff.unsqueeze(1) # (batch_size, 1)


class LookbackCall(BaseOption):
    """
    Floating lookback call option: payoff = S_T - min(S)
    Corresponds to Proposition 9.5 in the notes.
    """

    def __init__(self, idx_traded: List[int] = None):
        self.idx_traded = idx_traded

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d)
        Returns
        -------
        payoff: torch.Tensor
            Tensor of shape (batch_size, 1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded], 2)
        else:
            basket = torch.sum(x, 2)
        payoff = basket[:, -1] - torch.min(basket, 1)[0]   # S_T - min(S)
        return payoff.unsqueeze(1)


class BarrierOption(BaseOption):
    """
    Barrier option (knock-in or knock-out, call or put, up or down).
    Corresponds to Table 8.1 in the notes.

    Parameters
    ----------
    K : float
        Strike price
    B : float
        Barrier level
    option_type : str
        'call' or 'put'
    barrier_type : str
        'down-out', 'down-in', 'up-out', 'up-in'
    """

    def __init__(self, K: float, B: float, option_type: str = 'call', barrier_type: str = 'down-out'):
        self.K = K
        self.B = B
        self.option_type = option_type
        self.barrier_type = barrier_type

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d)
        Returns
        -------
        payoff: torch.Tensor
            Tensor of shape (batch_size, 1)
        """
        S = x[:, :, 0]          # use first asset, shape (batch_size, N)
        S_T = S[:, -1]          # terminal price, shape (batch_size,)

        # vanilla payoff at maturity
        if self.option_type == 'call':
            vanilla = torch.clamp(S_T - self.K, min=0.0)
        else:
            vanilla = torch.clamp(self.K - S_T, min=0.0)

        # check barrier crossing along the entire path
        if 'down' in self.barrier_type:
            crossed = torch.min(S, dim=1)[0] <= self.B   # (batch_size,)
        else:
            crossed = torch.max(S, dim=1)[0] >= self.B

        if 'out' in self.barrier_type:
            payoff = torch.where(crossed, torch.zeros_like(vanilla), vanilla)
        else:  # knock-in: only pays if barrier was crossed
            payoff = torch.where(crossed, vanilla, torch.zeros_like(vanilla))

        return payoff.unsqueeze(1)


class EuropeanCall(BaseOption):
    
    def __init__(self, K):
        """
        Parameters
        ----------
        K: float or torch.tensor
            Strike. Id K is a tensor, it needs to have shape (batch_size)
        """
        self.K = K

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, d) 
        Returns
        -------
        payoff: torch.Tensor
            basket option payoff. Tensor of shape (batch_size,1)
        """
        if x.dim()==3:
            return torch.clamp(x[:,-1,0]-self.K, 0).unsqueeze(1) # (batch_size, 1)
        elif x.dim() == 2:
            return torch.clamp(x[:,0]-self.K, 0).unsqueeze(1) # (batch_size, 1)
        else:
            raise ValueError('x needs to be last spot price, or trajectory of prices')
