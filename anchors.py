from pathlib import Path
import os,ast,copy
import dreamidv_wan.wan_swapface as wm
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv'
R=B/'B_5min_run01'
C=B.parent/'checkpoints/CP01_before_hair_AB_20261030'
def custom_generate():
    tree = ast.parse((C / 'snapshot/dreamidv/repo/dreamidv_wan/wan_swapface.py').read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DreamIDV'))
    fn = copy.deepcopy(next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')))
    fn.name = 'generate_anchored'
    injected = ast.parse("\nsigma_next=sample_scheduler.sigmas[_+1].to(device=device,dtype=torch.float32)\nassert 0 <= float(sigma_next) <= 1\nknown_next=(1-sigma_next)*latents_ref_video.float()+sigma_next*noise[0]\ntemp_x0=self._edit_latent*temp_x0+(1-self._edit_latent)*known_next.unsqueeze(0)\nprotected=(self._edit_latent==0).expand_as(temp_x0)\nerror=float((temp_x0-known_next.unsqueeze(0)).abs()[protected].max())\nassert error==0,error\nself._anchor_stats.append({'step':int(_),'sigma_next':float(sigma_next),'protected_latent_max_error':error})\n").body

    class Inject(ast.NodeTransformer):

        def __init__(self):
            self.count = 0

        def visit_Assign(self, node):
            if any((isinstance(t, ast.Name) and t.id == 'temp_x0' for t in node.targets)):
                self.count += 1
                return [node, *copy.deepcopy(injected)]
            return node
    transform = Inject()
    fn = transform.visit(fn)
    assert transform.count == 1
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    (R / 'custom_generate.py').write_text(ast.unparse(mod))
    namespace = dict(wm.__dict__)
    exec(compile(mod, 'hair_latent_anchor', 'exec'), namespace)
    return namespace['generate_anchored']
