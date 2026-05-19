define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global factorial
        .type   factorial, @function
factorial:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        str     x0, [fp, 16]  // n
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 32]  // spill lhs
        mov     x0, 1
        ldr     x9, [fp, 32]  // reload lhs
        cmp     x9, x0
        cset    x0, le
        cbz     x0, .Lif_end_0
        mov     x0, 1
        b       .Lfactorial_epilogue
.Lif_end_0:
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 32]  // spill lhs
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 40]  // spill lhs
        mov     x0, 1
        ldr     x9, [fp, 40]  // reload lhs
        sub     x0, x9, x0
        str     x0, [fp, 24]  // arg 0
        ldr     x0, [fp, 24]
        bl      factorial
        ldr     x9, [fp, 32]  // reload lhs
        mul     x0, x9, x0
.Lfactorial_epilogue:
        ldp     fp, lr, [sp], 48
        ret

        .balign 4
        .global main
        .type   main, @function
main:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        mov     x0, 5
        str     x0, [fp, 24]  // arg 0
        ldr     x0, [fp, 24]
        bl      factorial
        str     x0, [fp, 16]  // f
.Lmain_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 32
        ret
