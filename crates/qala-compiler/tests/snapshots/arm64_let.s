define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global let_demo
        .type   let_demo, @function
let_demo:
        stp     fp, lr, [sp, -64]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 48]  // spill lhs
        mov     x0, 1
        ldr     x9, [fp, 48]  // reload lhs
        add     x0, x9, x0
        str     x0, [fp, 24]  // x
        ldr     x0, [fp, 24]  // x
        str     x0, [fp, 48]  // spill lhs
        mov     x0, 2
        ldr     x9, [fp, 48]  // reload lhs
        mul     x0, x9, x0
        str     x0, [fp, 32]  // y
        ldr     x0, [fp, 32]  // y
        str     x0, [fp, 48]  // spill lhs
        ldr     x0, [fp, 16]  // a
        ldr     x9, [fp, 48]  // reload lhs
        sub     x0, x9, x0
        str     x0, [fp, 40]  // x
        ldr     x0, [fp, 40]  // x
.Llet_demo_epilogue:
        ldp     fp, lr, [sp], 64
        ret
