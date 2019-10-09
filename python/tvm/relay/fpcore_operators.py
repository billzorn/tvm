def compiler_relu(shape):
    ndem=len(shape)
    ident='relu'+str(ndem)
    args=[['t',*map(str,shape)]]
    return ['FPCore', ident, args,
            ['tensor',[('i' + str(i), str(x)) for i, x in enumerate(shape)],
             ['let', [('alt', ['get', 't', *('i' + str(i) for i in range(len(shape)))], ['if',['>','alt',0],'alt',0])]]]]

