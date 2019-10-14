def flatten1(matrix):
    return [elem for row in matrix for elem in row]

def unvar(s_expr):
    if isinstance(s_expr, (str, int)):
        return s_expr
    elif isinstance(s_expr, Expr):
        return unvar(s_expr.contents)
    elif isinstance(s_expr, (list, tuple)):
        return list(map(unvar, s_expr))
    else:
        raise TypeError(f'{s_expr!r} ({type(s_expr)}) is not valid')

def FPCore(name, args, body, properties=()):
    properties = properties if properties else []
    return unvar(['FPCore', name, *[arg.arg() for arg in args], *properties, body])

def while_(cond, vars_, tail):
    return Expr(['while', cond, flatten1(vars_), tail])

def while_var(var, initial, update):
    return [var, initial, update]

class Expr:
    def __init__(self, contents):
        self.contents = contents
    def __add__(self, other):
        return Expr(['+', self, other])
    def __lt__(self, other):
        return Expr(['<', self, other])

class Var(Expr):
    def __init__(self, name):
        self.name = name
    @property
    def contents(self):
        return self.name
    def arg(self):
        return self.name

class TensorVar(Var):
    def __init__(self, name, dims, shape=None):
        self.name = name
        self.dims = dims
        self.shape = shape
    def create(self, body):
        return Expr(['tensor', [[self % i, self.shape[i]] for i in range(self.dims)], body])
    def __getitem__(self, indices):
        return Expr(['get', self.name, *indices])
    def __mod__(self, dim):
        '''A variable representing the ith dimension of this tensor.

Exists when this tensor is being created, or when this tensor is being used as an arg to a function.'''
        return Var(self.name + '_' + str(dim))
    def arg(self):
        if self.shape is None:
            return Expr([self, [self % dim for dim in range(self.dims)]])
        else:
            return Expr([self, self.shape])
    @property
    def contents(self):
        return self.name
