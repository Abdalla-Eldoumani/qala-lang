define(fp, x29)
define(lr, x30)

        .data
.Lfmt_0:  .string "square(%lld) = %lld\n"

        .text

        .balign 4
        .global square
        .type   square, @function
square:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        str     x0, [fp, 16]  // n
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 24]  // spill lhs
        ldr     x0, [fp, 16]  // n
        ldr     x9, [fp, 24]  // reload lhs
        mul     x0, x9, x0
.Lsquare_epilogue:
        ldp     fp, lr, [sp], 32
        ret

        .balign 4
        .global main
        .type   main, @function
main:
        stp     fp, lr, [sp, -48]!
        mov     fp, sp
        mov     x0, 6
        str     x0, [fp, 16]  // x
        ldr     x0, [fp, 16]  // x
        str     x0, [fp, 24]  // printf arg 1
        ldr     x0, [fp, 16]  // x
        str     x0, [fp, 32]  // arg 0
        ldr     x0, [fp, 32]
        bl      square
        str     x0, [fp, 32]  // printf arg 2
        ldr     x1, [fp, 24]
        ldr     x2, [fp, 32]
        ldr     x0, =.Lfmt_0  // printf format
        bl      printf
.Lmain_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 48
        ret
