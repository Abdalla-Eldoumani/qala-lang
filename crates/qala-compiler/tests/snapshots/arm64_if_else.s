define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global sign
        .type   sign, @function
sign:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        str     x0, [fp, 16]  // n
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 24]  // spill lhs
        mov     x0, 0
        ldr     x9, [fp, 24]  // reload lhs
        cmp     x9, x0
        cset    x0, eq
        cbz     x0, .Lif_end_0
        mov     x0, 0
        b       .Lsign_epilogue
.Lif_end_0:
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 24]  // spill lhs
        mov     x0, 0
        ldr     x9, [fp, 24]  // reload lhs
        cmp     x9, x0
        cset    x0, gt
        cbz     x0, .Lif_else_2
        mov     x0, 1
        b       .Lsign_epilogue
        b       .Lif_end_1
.Lif_else_2:
        mov     x0, 1
        neg     x0, x0
        b       .Lsign_epilogue
.Lif_end_1:
        mov     x0, 0
.Lsign_epilogue:
        ldp     fp, lr, [sp], 32
        ret

        .balign 4
        .global grade
        .type   grade, @function
grade:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        str     x0, [fp, 16]  // s
        ldr     x0, [fp, 16]  // s
        str     x0, [fp, 24]  // spill lhs
        mov     x0, 90
        ldr     x9, [fp, 24]  // reload lhs
        cmp     x9, x0
        cset    x0, ge
        cbz     x0, .Lif_else_4
        mov     x0, 4
        b       .Lgrade_epilogue
        b       .Lif_end_3
.Lif_else_4:
        ldr     x0, [fp, 16]  // s
        str     x0, [fp, 24]  // spill lhs
        mov     x0, 80
        ldr     x9, [fp, 24]  // reload lhs
        cmp     x9, x0
        cset    x0, ge
        cbz     x0, .Lif_else_6
        mov     x0, 3
        b       .Lgrade_epilogue
        b       .Lif_end_5
.Lif_else_6:
        mov     x0, 2
        b       .Lgrade_epilogue
.Lif_end_5:
.Lif_end_3:
        mov     x0, 0
.Lgrade_epilogue:
        ldp     fp, lr, [sp], 32
        ret
