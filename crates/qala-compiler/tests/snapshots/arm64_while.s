define(fp, x29)
define(lr, x30)

        .text

        .balign 4
        .global loop_demo
        .type   loop_demo, @function
loop_demo:
        stp     fp, lr, [sp, -32]!
        mov     fp, sp
        str     x0, [fp, 16]  // c
        b       .Lwhile_test_1
.Lwhile_body_0:
        ldr     x0, [fp, 16]  // c
        cbz     x0, .Lif_end_3
        b       .Lwhile_end_2
.Lif_end_3:
        b       .Lwhile_test_1
.Lwhile_test_1:
        ldr     x0, [fp, 16]  // c
        cbnz    x0, .Lwhile_body_0
.Lwhile_end_2:
.Lloop_demo_epilogue:
        mov     x0, 0  // void return -> exit code 0
        ldp     fp, lr, [sp], 32
        ret
