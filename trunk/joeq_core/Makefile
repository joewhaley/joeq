packages = Allocator Assembler Bootstrap ClassLib Clazz Compil3r Interpreter Linker Main Run_Time Scheduler Synchronization UTF Util jq.java

all:	jikes

jikes:
	find ${packages} -name "*.java" | xargs jikes

jikes-pedantic:
	find ${packages} -name "*.java" | xargs jikes -P

javac:
	find ${packages} -name "*.java" | xargs javac

clean:
	find ${packages} -name "*.class" | xargs rm

update:
	( export CVS_RSH=ssh ; cvs update -P )

commit:
	( export CVS_RSH=ssh ; cvs commit )

bootstrap:
	java -mx200M Main.Bootstrapper
