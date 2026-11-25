"""SAE order interventions in model units, with a symmetric in-domain mask."""
import torch


def decompose(sae, donor, recipient):
    if donor.shape != recipient.shape or donor.ndim != 2:
        raise ValueError('expected aligned [tokens, hidden] states')
    device = sae.W_dec.device
    d, r = donor.to(device), recipient.to(device)
    eligible = ~(sae.is_massive(d) | sae.is_massive(r))
    # Do not encode excluded tokens, even for descriptive feature statistics.
    zd, zr = sae.encode(d[eligible]), sae.encode(r[eligible])
    ad, ar = sae.decode(zd), sae.decode(zr)
    reconstruction = torch.zeros_like(r)
    reconstruction[eligible] = ad - ar
    residual = torch.zeros_like(r)
    residual[eligible] = (d[eligible] - ad) - (r[eligible] - ar)
    delta = d-r
    outside = delta.clone(); outside[eligible] = 0
    error = (delta-reconstruction-residual-outside).abs().max().item()
    if error > 1e-3:
        raise ValueError(f'decomposition failed: {error}')
    return dict(eligible=eligible.cpu(), reconstructed_delta=reconstruction.cpu(),
                residual_delta=residual.cpu(), outside_delta=outside.cpu(),
                donor_reconstruction=ad.cpu(), recipient_reconstruction=ar.cpu(),
                donor_latents=zd.cpu(), recipient_latents=zr.cpu(), algebra_error=error)


def moments(h, reconstruction):
    h, reconstruction = h.double(), reconstruction.double()
    return dict(n=len(h), sse=float((h-reconstruction).square().sum()),
                sum=h.sum(0).tolist(), sumsq=float(h.square().sum()))