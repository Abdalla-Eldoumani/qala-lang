define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global add3
        .type   add3, @function
add3:
        stp     fp, lr, [sp, -64]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        str     x2, [fp, 32]  // c
        ldr     x0, [fp, 16]  // a
        str     x0, [fp, 48]  // spill lhs
        ldr     x0, [fp, 24]  // b
        ldr     x9, [fp, 48]  // reload lhs
        add     x0, x9, x0
        str     x0, [fp, 40]  // sum
        ldr     x0, [fp, 40]  // sum
        str     x0, [fp, 48]  // spill lhs
        ldr     x0, [fp, 32]  // c
        ldr     x9, [fp, 48]  // reload lhs
        add     x0, x9, x0
.Ladd3_epilogue:
        ldp     fp, lr, [sp], 64
        ret

        .balign 4
        .global main
        .type   main, @function
main:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        mov     x0, 10
        str     x0, [fp, 24]  // arg 0
        mov     x0, 20
        str     x0, [fp, 32]  // arg 1
        mov     x0, 12
        str     x0, [fp, 40]  // arg 2
        ldr     x0, [fp, 24]
        ldr     x1, [fp, 32]
        ldr     x2, [fp, 40]
        bl      add3
        str     x0, [fp, 16]  // r
.Lmain_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 48
        ret
