import torch

# Uncertainty Estimation for Models to select

class UncertaintyScore():
    def __init__(self,type):
        """
        :param type: Type of sampling policy
        """
        self.type = type

    def LeastConfidence(self,logits):
        """
        :param logits: logits of the sample
        :return: uncertainty score for each sample
        """
        return 1 - torch.max(torch.softmax(logits)) 

    def MarginSampling(self,logits):
        """
        :param logits: logits of the sample
        :return: uncertainty score for each sample
        """
        return torch.sort(torch.softmax(logits))[0][-1] - torch.sort(torch.softmax(logits))[0][-2]

    def Entropy(self,logits):
        """
        :param logits: logits of the sample
        :return: uncertainty score for each sample
        """
        return -torch.sum(torch.softmax(logits) * torch.log(torch.softmax(logits)))
    
    def __call__(self,logits):
        """
        :param logits: logits of the sample
        :return: uncertainty score for each sample
        """
        if self.type == "LeastConfidence":
            return self.LeastConfidence(logits)
        elif self.type == "MarginSampling":
            return self.MarginSampling(logits)
        elif self.type == "Entropy":
            return self.Entropy(logits)
        else:
            raise ValueError("Invalid Sampling Policy")