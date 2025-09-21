	.file	"demo.c"
	.text
	.section .rdata,"dr"
.LC1:
	.ascii "%d%c\0"
.LC0:
	.ascii " \12\0"
	.text
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
	.seh_proc	main
main:
	pushq	%rbp
	.seh_pushreg	%rbp
	movq	%rsp, %rbp
	.seh_setframe	%rbp, 0
	subq	$64, %rsp
	.seh_stackalloc	64
	.seh_endprologue
	call	__main
	movl	$1, -32(%rbp)
	movl	$2, -28(%rbp)
	movl	$3, -24(%rbp)
	movl	$4, -20(%rbp)
	movl	$5, -16(%rbp)
	movl	$5, -8(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L2
.L3:
	movl	-8(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -4(%rbp)
	sete	%al
	movzbl	%al, %eax
	cltq
	leaq	.LC0(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %ecx
	movl	-4(%rbp), %eax
	cltq
	movl	-32(%rbp,%rax,4), %edx
	leaq	.LC1(%rip), %rax
	movl	%ecx, %r8d
	movq	%rax, %rcx
	call	printf
	addl	$1, -4(%rbp)
.L2:
	movl	-4(%rbp), %eax
	cmpl	-8(%rbp), %eax
	jl	.L3
	movl	$0, %ecx
	call	exit
	nop
	.seh_endproc
	.def	__main;	.scl	2;	.type	32;	.endef
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 15.2.0"
	.def	printf;	.scl	2;	.type	32;	.endef
	.def	exit;	.scl	2;	.type	32;	.endef
