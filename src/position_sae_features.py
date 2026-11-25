"""Feature-value transplants with residual preservation, and paired first-order prediction."""
import torch


def feature_delta(sae,donor_z,recipient_z,features):
    if donor_z.shape!=recipient_z.shape:raise ValueError('latent shape mismatch')
    if len(set(features))!=len(features):raise ValueError('duplicate feature')
    if any(i<0 or i>=sae.n_latents for i in features):raise ValueError('feature out of range')
    device=sae.W_dec.device
    dz=(donor_z[:,features]-recipient_z[:,features]).to(device)
    return (dz @ sae.W_dec[features] / sae.scale).cpu()


def order_gradient(runner,case,qid,q,donor,recipient):
    """Gradient of clean order-contrast dot logits through the frozen marker-only path."""
    from src.activations import capture
    from run_position_path import patch_hook
    # Predict only obtains the exact official input tensors; replay them with gradients.
    import copy
    qq=copy.deepcopy(q);qq['option_order']=recipient['order']
    with capture(runner.agent,[]) as cap:
        runner.agent.predict(case['state'],{qid:qq},max_len=1024,head_max_len=256)
    args=[x.detach().clone().to(runner.agent.device) for x in cap.forward_calls[0][0][:5]]
    holder={}
    def leaf(_m,_a,output):
        h=output[0] if isinstance(output,tuple) else output
        x=h.detach().clone().requires_grad_(True);holder['h']=x
        return (x,)+tuple(output[1:]) if isinstance(output,tuple) else x
    handles=[runner.modules['enc_17'].register_forward_hook(leaf),
        runner.modules['enc_18'].register_forward_hook(patch_hook(recipient['alignment']['indices'],recipient['states']['enc_18'],recipient['alignment']['groups']['nonmarkers']))]
    try:
        with torch.enable_grad():
            logits,_=runner.agent.model(*args)
            aligned=logits[0,[recipient['order'].index(i) for i in range(len(recipient['order']))]].float()
            error=float((aligned.detach().cpu()-torch.tensor(recipient['logits'])).abs().max())
            if error>1e-4:raise ValueError(f'gradient replay does not reproduce baseline: {error}')
            contrast=torch.tensor(donor['logits'],device=aligned.device)-torch.tensor(recipient['logits'],device=aligned.device)
            contrast-=contrast.mean()
            objective=(aligned*contrast).sum()
            gradient=torch.autograd.grad(objective,holder['h'])[0]
        return gradient[0,recipient['alignment']['indices']].detach().cpu(),error
    finally:
        for handle in handles:handle.remove()