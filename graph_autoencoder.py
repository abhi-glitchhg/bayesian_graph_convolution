from torch_geometric.nn  import BatchNorm
from torch_geometric.utils import (add_self_loops, negative_sampling, remove_self_loops)
from .bayesian_gcn import BGCNConv
import torch.nn as nn
import torch

EPS = 1e-5


class BN_GAE(nn.Module):

    def __init__(self, encoder, decoder,stddev_prior=0.1):
        super().__init__()
        self.stddev_prior= stddev_prior
        self.encoder = encoder
        self.decoder = decoder

    def get_pw(self,):
        return self.encoder.get_pw() # + self.decoder.get_pw()
    
    def get_qw(self,):
        return self.encoder.get_qw() #+ self.decoder.get_qw()

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, input,nb_samples,neg_edge_index=None):
      
      total_loss,total_qw, total_pw, total_log_likelihood = 0., 0., 0., 
      for sample_ in range(nb_samples):
        output =self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(output + EPS).mean()
        total_qw += self.get_qw()
        total_pw += self.get_pw()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        total_loss +=neg_loss + pos_loss

        #adding kl divergence loss
        
      #print("loss because of the prediction : ",total_loss, " losses related to the kl-dicergence without normalization :",total_qw- total_pw, "are the losses :(")
      return {"loss":total_loss/nb_samples,"total_qw":total_qw/nb_samples,"total_pw":total_pw/nb_samples }

    def test(self, z, pos_edge_index, neg_edge_index):
 
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        #print(y, "is y", pred," is pred")
        return roc_auc_score(y, pred), average_precision_score(y, pred)