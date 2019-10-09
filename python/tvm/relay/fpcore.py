import random
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


class GenerateFPCore(ExprMutator):
    def node_name(self, node):
        # returns a unique name for each node
        return str(random.randint(int(1e4), int(1e5)))

    def visit_function(self, function):
        return [
            'FPCore',
            'fn-' + self.node_name(function),
            [self.visit_param(param) for param in function.params],
            self.visit(function.body),
        ]

    def visit_let(self, let):
        return [
            'let',
            [self.visit_param(let.var), self.visit(let.value)],
            self.visit(let.body),
        ]

    def visit_call(self, call):
        return [
            self.visit_op(call.op),
            *[self.visit(arg) for arg in call.args],
        ]

    def visit_param(self, var):
        # this is var name that is being DEFINED
        return 'v-' + self.node_name(var)

    def visit_var(self, var):
        # this is var name that is REFERENCED
        return 'v-' + self.node_name(var)

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
        # Convert this op to an FPCore op
        return str(op)

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
