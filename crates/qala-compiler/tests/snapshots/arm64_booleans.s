define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global andor
        .type   andor, @function
andor:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        str     x0, [fp, 16]  // a
        str     x1, [fp, 24]  // b
        ldr     x0, [fp, 16]  // a
        cbz     x0, .Land_false_2
        ldr     x0, [fp, 24]  // b
        cbz     x0, .Land_false_2
        mov     x0, 1
        b       .Land_done_3
.Land_false_2:
        mov     x0, 0
.Land_done_3:
        cbnz    x0, .Lor_true_0
        ldr     x0, [fp, 16]  // a
        eor     x0, x0, 1
        cbnz    x0, .Lor_true_0
        mov     x0, 0
        b       .Lor_done_1
.Lor_true_0:
        mov     x0, 1
.Lor_done_1:
.Landor_epilogue:
        ldp     fp, lr, [sp], 32
        ret
