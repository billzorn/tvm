from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.ty import TensorType


def infer_type(func):
    mod = relay.Module.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod["main"]


class GenerateFPCore(ExprMutator):
    def visit_function(self, function):
        return [
            'FPCore',
            'fn-' + str(hash(function)),
            [self.visit_function_param(param) for param in function.params],
            self.visit(function.body),
        ]

    def visit_function_param(self, var):
        type_ = var.checked_type
        if type(type_) is not TensorType:
            raise NotImplementedError('Not implemented')
        return [
            '!', ':tvm-type', type_.dtype,
            self.visit_var(var),
            *type_.shape,
        ]

    def visit_let(self, let):
        return [
            'let',
            [self.visit_var(let.var), self.visit(let.value)],
            self.visit(let.body),
        ]

    def visit_call(self, call):
        # [arg.checked_type for arg in call.args]
        return [
            self.visit_op(call.op) + '-' + ','.join(['x'.join(map(str, arg.checked_type.shape)) for arg in call.args]),
            *[self.visit(arg) for arg in call.args],
        ]

    def visit_var(self, var):
        return 'var-' + str(var.vid) + '-' + 'x'.join(map(str, var.checked_type.shape))

    def visit_type(self, type_):
        raise NotImplementedError(f'{type_} ({type(type_)}) is not supported')

    def visit_if(self, if_):
        return [
            'if',
            self.visit(if_.cond),
            self.visit(if_.true_branch),
            self.visit(if_.false_branch),
        ]

    def visit_tuple(self, tuple_):
        raise NotImplementedError(f'{tuple_} ({type(tuple_)}) is not supported')

    def visit_tuple_getitem(self, tg):
        raise NotImplementedError(f'{tg} ({type(tg)}) is not supported')

    def visit_global_var(self, gvar):
        raise NotImplementedError(f'{gvar} ({type(gvar)}) is not supported')

    def visit_op(self, op):
        return str(op.name)

    def visit_constant(self, constant):
        return str(constant)

    def visit_ref_create(self, ref_create):
        raise NotImplementedError(f'{ref_create} ({type(ref_create)}) is not supported')

    def visit_ref_write(self, ref_write):
        raise NotImplementedError(f'{ref_write} ({type(ref_write)}) is not supported')

    def visit_ref_read(self, ref_read):
        raise NotImplementedError(f'{ref_read} ({type(ref_read)}) is not supported')

    def visit_constructor(self, constructor):
        raise NotImplementedError(f'{constructor} ({type(constructor)}) is not supported')

    def visit_match(self, match):
        raise NotImplementedError(f'{match} ({type(match)}) is not supported')
