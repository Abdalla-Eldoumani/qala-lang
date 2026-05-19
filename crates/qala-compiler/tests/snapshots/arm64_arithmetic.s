define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global arith
        .type   arith, @function
arith:
        stp     fp, lr, [sp, -64]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 56]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 56]  // reload lhs
        add     x0, x9, x0
        str     x0, [fp, 48]  // spill lhs
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 56]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 56]  // reload lhs
        sub     x0, x9, x0
        ldr     x9, [fp, 48]  // reload lhs
        mul     x0, x9, x0
        str     x0, [fp, 40]  // spill lhs
        mov     x0, 2
        ldr     x9, [fp, 40]  // reload lhs
        sdiv    x0, x9, x0
        str     x0, [fp, 32]  // spill lhs
        mov     x0, 7
        ldr     x9, [fp, 32]  // reload lhs
        sdiv    x10, x9, x0
        msub    x0, x10, x0, x9
.Larith_epilogue:
        ldp     fp, lr, [sp], 64
        ret
