from .fpcore_lang import *

def compiler_relu(shape):
    ndem=len(shape)
    ident='relu-'+str(ndem)
    args=[['t',*map(str,shape)]]
    return ['FPCore', ident, args,
            ['tensor',[('i' + str(i), str(x)) for i, x in enumerate(shape)],
             ['let', [('alt', ['get', 't', *('i' + str(i) for i in range(len(shape)))], ['if',['>','alt',0],'alt',0])]]]]

def dense():
    A = TensorVar('A', 2)
    B = TensorVar('B', 2)
    C = TensorVar('C', 2, (A%1, B%1))
    S = Var('S')
    i, j, k = Var('i'), Var('j'), Var('k')
    return FPCore(
        name='dense',
        args=(A, B),
        body=C.create(
            while_(
                cond=k < A%2,
                vars_=[
                    while_var(var=S, initial=0, update=A[i, k] + B[j, k])
                ],
                tail=S,
            )
        )
    )


if __name__ == '__main__':
    print(compiler_relu((2, 3, 4)))
    print(dense())
