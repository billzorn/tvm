from tvm import relay
from .transform import function_pass


@function_pass(opt_level=1)
class FPCorePass:
    """
    Change the batch size.

    Parameters
    ----------

    Returns
    -------
    pass: FunctionPass
      The pass.
    """
    def __init__(self):
        pass

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        sself = self
        class FPCorePassFunctor(relay.ExprVisitor):
            pass
        return FPCorePassFunctor().visit(func)
