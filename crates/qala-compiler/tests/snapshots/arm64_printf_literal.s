define(fp, x29)
define(lr, x30)

        .data
.Lfmt_0:  .string "done\n"

        .text

        .balign 4
        .global main
        .type   main, @function
main:
        stp     fp, lr, [sp, -16]!
        mov     fp, sp
        ldr     x0, =.Lfmt_0  // printf format
        bl      printf
.Lmain_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 16
        ret
