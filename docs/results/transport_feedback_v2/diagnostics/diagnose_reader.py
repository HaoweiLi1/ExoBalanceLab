import json,time,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
from exobalancelab.transport_policy import TransportPolicy
from exobalancelab.simulation import RunConfig,run
from exobalancelab.reachability import PolicyCoverageError
from examples.validate_transport_feedback import terminal_outcome,write
class SwitchingPolicy(TransportPolicy):
 def __init__(self,path):
  super().__init__(path); self.rates=np.array([1.,-1.])
 def rate(self,remaining_s,lean_rad,velocity_rad_s,human_torque_nm,profile):
  sign=super().rate(remaining_s,lean_rad,velocity_rad_s,human_torque_nm,profile)
  return profile.rate_min_nm_s if sign>0 else profile.rate_max_nm_s if sign<0 else 0.
root=Path('logs/transport_feedback');out=root/'diagnostics';out.mkdir()
write(out/'protocol.json',dict(cases=[['OM',1],['OM',5],['YF',3]],changes=['Same bounded-rate interpolation with 0.5 ms control','Hamiltonian extremum of interpolated unit switching function at original 1 ms control'],acceptance='Diagnostic only; original 1 ms campaign and all criteria retained.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
results=[]
for profile,index in [('OM',1),('OM',5),('YF',3)]:
 name=next(p.name for p in (root/'policies').iterdir() if p.is_dir() and p.name.startswith(profile+'_none_'))
 source=json.loads((root/'policies'/name/'summary.json').read_text())
 case=json.loads((root/'rollouts_v2'/f'selection_{profile}.json').read_text())['samples'][index]
 for mode,period,kind in [('faster',.0005,TransportPolicy),('switching',.001,SwitchingPolicy)]:
  c=RunConfig(model='paper',profile=profile,parameters='authors',exo='none',duration_s=1.4,timestep_s=.00000625,control_period_s=period,lean_rad=case['state'][0]-np.pi/2,velocity_rad_s=case['state'][1],initial_torque_nm=case['state'][2])
  r=dict(profile=profile,index=index,mode=mode,config=asdict(c),state=case['state'])
  with kind(root/'policies'/name/'policy.h5') as p:
   r['policy_sha256']=p.sha256
   try:e=run(c,hjb=p)
   except (ValueError,RuntimeError) as error:
    r.update(status='rollout_error',error=str(error))
    if isinstance(error,PolicyCoverageError):r['failed_query']=error.query
   else:
    r.update(status='completed',metrics=e.metrics,**terminal_outcome(e,source['target']))
    e.arrays={k:v[::160] for k,v in e.arrays.items()};e.save(out/f'{profile}_{index}_{mode}')
  results.append(r);write(out/'results.json',results);print(profile,index,mode,r['status'],r.get('valid_target_reach'),r.get('error'),flush=True)
