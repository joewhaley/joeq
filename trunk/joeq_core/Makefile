packages = Allocator Assembler Bootstrap ClassLib Clazz Compil3r Interpreter Linker Main Run_Time Scheduler Synchronization UTF Util java

all:	jikes

jikes:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -bootclasspath @bootclasspath

jikes-pedantic:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -bootclasspath @bootclasspath -P

javac:
	find ${packages} -name "*.java" | xargs javac

clean:
	find ${packages} -name "*.class" | xargs rm -f

veryclean:
	find ${packages} -name "*.class" | xargs rm -f bootclasspath

update:
	( export CVS_RSH=ssh ; cvs update -Pd )

commit:
	( export CVS_RSH=ssh ; cvs commit )

bootstrap:
	java -Xbootclasspath/a:. -mx250M Main.Bootstrapper

bootclasspath:	Main/GetBootClassPath.class
	java Main.GetBootClassPath > bootclasspath

Main/GetBootClassPath.class:	Main/GetBootClassPath.java
	javac Main/GetBootClassPath.java
