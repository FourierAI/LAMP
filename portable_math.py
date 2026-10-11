"""Complete FP32 systems, CUDA arithmetic; host/mmap stores large dense Grams."""
from pathlib import Path
import torch
import block_lu
import kernels

_OPTIONS = dict(dense_limit=12000, tile=2048, scratch=None)

def configure(output, dense_limit=12000, tile=2048):
    _OPTIONS.update(dense_limit=dense_limit,tile=tile,scratch=Path(output)/'scratch')
    _OPTIONS['scratch'].mkdir(parents=True,exist_ok=True)

def path(name): return _OPTIONS['scratch']/(name+'.bin')

def global_gram(x,beta):
    if len(x) <= _OPTIONS['dense_limit']: return kernels.global_kernel(x,x,beta)
    return block_lu.global_gram(x,beta,path('gram'))

def region_gram(x,beta):
    if len(x) <= _OPTIONS['dense_limit']: return kernels.set_kernels(x,x,[beta],128,128,symmetric=True)[0]
    return block_lu.native_gram(x,beta,kernels.set_kernels,path('gram'))

def center_support(gram):
    if not torch.is_tensor(gram):
        state,_ = block_lu.center_support(gram,torch.device('cuda',torch.cuda.current_device()))
        return gram,state
    means=gram.sum(1,dtype=torch.float64)/len(gram)
    mean=means.mean();variance=gram.diagonal().double()-2*means+mean
    if float(variance.min()) < -2e-6: raise RuntimeError('Negative support variance')
    scale=variance.clamp_min(1e-10).sqrt()
    result=torch.empty_like(gram)
    for begin in range(0,len(gram),512):
        end=min(begin+512,len(gram));value=gram[begin:end].double()-means[begin:end,None]-means[None,:]+mean
        result[begin:end]=(value/(scale[begin:end,None]*scale[None,:])).float()
    return result,dict(means=means,mean=mean,scale=scale)

def solve(gram,rhs,lam):
    if torch.is_tensor(gram): return kernels.solve(gram,rhs,lam)
    work=block_lu.coefficient_store(gram,lam,rhs.device,path=path('work'),dtype=torch.float32)
    result,record=block_lu.factor_solve(work,rhs,rhs.device,tile=_OPTIONS['tile'])
    del work
    path('work').unlink(missing_ok=True)
    check=block_lu.residual(gram,lam,rhs,result)
    if check['backward_error'] >= 1e-6: raise RuntimeError('Unstable blocked direct solve')
    return result

def numerical_residual(gram,rhs,alpha,lam):
    if torch.is_tensor(gram): return kernels.numerical_residual(gram,rhs,alpha,lam)
    return block_lu.residual(gram,lam,rhs,alpha)

def global_response(q,s,alpha,beta):
    if len(s) <= _OPTIONS['dense_limit']: return kernels.global_kernel(q,s,beta)@alpha
    return torch.cat([kernels.global_kernel(q[start:start+512],s,beta)@alpha for start in range(0,len(q),512)])
