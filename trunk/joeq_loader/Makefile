all:	set_text_writeable
	gcc -g joeq.c native.c exception.c ../joeq/jq.obj -lpthread -o joeq
	./set_text_writeable joeq

set_text_writeable:	set_text_writeable.c
	gcc -g set_text_writeable.c -o set_text_writeable