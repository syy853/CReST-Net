"""Static, read-only source verification. Does not instantiate models or use a GPU."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
model = root / 'models' / 'CReSTNet.py'
modules = root / 'models' / 'modules.py'
checks = []

def check(name, condition):
    checks.append((name, bool(condition)))
    print(('[PASS] ' if condition else '[FAIL] ') + name)

check('Main model file renamed', model.is_file() and not (root / 'models' / 'GCoNet.py').exists())
model_tree = ast.parse(model.read_text())
classes = {n.name for n in model_tree.body if isinstance(n, ast.ClassDef)}
check('CReSTNet is the main model class', 'CReSTNet' in classes and 'GCoNet' not in classes)
for name in ('train.py', 'test.py'):
    src = (root/name).read_text()
    check(name + ' imports CReSTNet', 'from models.CReSTNet import CReSTNet' in src)
    check(name + ' instantiates CReSTNet', 'CReSTNet()' in src)

module_tree = ast.parse(modules.read_text())
module_classes = {n.name for n in module_tree.body if isinstance(n, ast.ClassDef)}
for name in ('GCAM','CoAttLayer','GWM','SGS','CPKA'):
    check('Legacy class removed: '+name, name not in module_classes)
for name in ('DiversePrototypeConsensusMining','ObjectnessVerifiedDCD'):
    check('Active class retained: '+name, name in module_classes)
for name in ('LateVariationStructurePreservation','ConsensusGuidedPromptFeedback','DPTDecoder','DINOv2PyramidAdapter'):
    check('Active component referenced: '+name, name in model.read_text())
check('Configuration unchanged/available', (root/'config.py').is_file())
print('\nSUMMARY:',sum(v for _,v in checks),'/',len(checks),'checks passed')
raise SystemExit(0 if all(v for _,v in checks) else 1)
