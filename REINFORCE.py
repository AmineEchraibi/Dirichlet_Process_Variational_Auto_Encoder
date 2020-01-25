import torch
eps = 1e-9

class ReinforceGradientNormal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, h, mu, invvar, input):
        ctx.save_for_backward(h, mu, invvar, input)
        output = input
        return output




    @staticmethod
    def backward(ctx, grad_output):
        h, mu, invvar, input = ctx.saved_tensors
        grad_h = None
        grad_mu = None
        grad_invvar = None
        grad_input = None


        if ctx.needs_input_grad[1]:
            grad_mu =  - ( h - mu) * invvar * input

        if ctx.needs_input_grad[2]:
            grad_invvar = 0.5 *(1 / invvar -(h - mu).pow(2)) * input

        if ctx.needs_input_grad[3]:

            grad_input = grad_output.clone()

        return grad_h, grad_mu, grad_invvar, grad_input



reinforce_normal = ReinforceGradientNormal.apply