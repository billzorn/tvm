import tvm
from tvm import relay
import numpy as np

m = relay.Module()

a = relay.var('foo', shape=[1,2,3], dtype="float64")
f = relay.Function([a], relay.const(2.0, dtype="float64") * a)
m['main'] = f

e = relay.create_executor(mod=m)
ev = e.evaluate()




print(ev(np.random.randn(1,2,3)))

def module_to_source(relay_mod, target):
    built = relay.build(relay_mod, target=target)
    tvmmodule = built[1]
    return tvmmodule.get_source()
