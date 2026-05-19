define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global for_demo
        .type   for_demo, @function
for_demo:
        stp     fp, lr, [sp, -64]!
        mov     fp, sp
        str     x0, [fp, 16]  // n
        mov     x0, 0
        str     x0, [fp, 24]  // i
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 32]  // for end
        b       .Lfor_test_2
.Lfor_body_0:
.Lfor_incr_1:
        ldr     x0, [fp, 24]
        add     x0, x0, 1
        str     x0, [fp, 24]
.Lfor_test_2:
        ldr     x0, [fp, 24]
        ldr     x9, [fp, 32]
        cmp     x0, x9
        b.lt    .Lfor_body_0
.Lfor_end_3:
        mov     x0, 0
        str     x0, [fp, 40]  // j
        ldr     x0, [fp, 16]  // n
        str     x0, [fp, 48]  // for end
        b       .Lfor_test_6
.Lfor_body_4:
.Lfor_incr_5:
        ldr     x0, [fp, 40]
        add     x0, x0, 1
        str     x0, [fp, 40]
.Lfor_test_6:
        ldr     x0, [fp, 40]
        ldr     x9, [fp, 48]
        cmp     x0, x9
        b.le    .Lfor_body_4
.Lfor_end_7:
.Lfor_demo_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 64
        ret
