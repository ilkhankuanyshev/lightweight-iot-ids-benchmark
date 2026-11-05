"""Restart and execute every notebook cell with the current Python interpreter."""
import os
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec
from jupyter_client.manager import KernelManager

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
runtime=Path(tempfile.gettempdir())/'iot-ids-jupyter'
runtime.mkdir(parents=True,exist_ok=True)
for key in ('IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR'):
    os.environ.setdefault(key,str(runtime/key.lower()))
class CurrentPythonSpec(KernelSpecManager):
    def get_kernel_spec(self,name):
        return KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                          display_name='Python 3',language='python')

path=root/'notebooks/01_data_analysis.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3',kernel_spec_manager=CurrentPythonSpec())
try:
    NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(root)}}).execute()
finally:
    if km.has_kernel:
        km.shutdown_kernel(now=True)
nbformat.write(nb,path)
counts=[c.execution_count for c in nb.cells if c.cell_type=='code']
assert counts==list(range(1,len(counts)+1)),counts
assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
print('Executed notebook:',counts)
