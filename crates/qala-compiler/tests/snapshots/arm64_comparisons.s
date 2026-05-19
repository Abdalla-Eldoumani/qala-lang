define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global cmp
        .type   cmp, @function
cmp:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 32]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 32]  // reload lhs
        cmp     x9, x0
        cset    x0, eq
.Lcmp_epilogue:
        ldp     fp, lr, [sp], 48
        ret

        .balign 4
        .global lt
        .type   lt, @function
lt:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 32]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 32]  // reload lhs
        cmp     x9, x0
        cset    x0, lt
.Llt_epilogue:
        ldp     fp, lr, [sp], 48
        ret

        .balign 4
        .global ge
        .type   ge, @function
ge:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 32]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 32]  // reload lhs
        cmp     x9, x0
        cset    x0, ge
.Lge_epilogue:
        ldp     fp, lr, [sp], 48
        ret
