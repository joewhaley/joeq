
all:	linux

mingw:	set_text_writeable
	gcc -mno-cygwin -g joeq.c native.c exception.c ../joeq/jq.obj -o joeq.exe
	./set_text_writeable joeq.exe

msvc:	set_text_writeable_msvc
	cl /DWIN32 joeq.c native.c exception.c ../joeq/jq.obj
	set_text_writeable.exe joeq.exe

linux:	set_text_writeable
	gcc -g joeq.c native.c exception.c ../joeq/jq.obj -lpthread -ldl -o joeq
	./set_text_writeable joeq

set_text_writeable:	set_text_writeable.c
	gcc -g set_text_writeable.c -o set_text_writeable

set_text_writeable_msvc:	set_text_writeable.c
	cl /DWIN32 set_text_writeable.c
